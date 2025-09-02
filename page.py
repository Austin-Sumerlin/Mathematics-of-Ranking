import re
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx
import streamlit as st

# ==========================
# Parsing helpers & graph build
# ==========================

def _split_csvish(line: str):
    """Split on commas with arbitrary spaces around them."""
    return re.split(r"\s*,\s*", line.strip())

@st.cache_data(show_spinner=False)
def load_team_names_from_text(text: str):
    """
    Input lines like: "10, Arizona" or "1, Abilene_Chr".
    Returns mappings id_to_name and name_to_id.
    """
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}

    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = _split_csvish(raw)
        if len(parts) < 2:
            continue
        try:
            tid = int(parts[0])
        except ValueError:
            continue
        name = re.sub(r"\s+", " ", parts[1].replace("_", " ")).strip()
        id_to_name[tid] = name
        name_to_id[name.upper()] = tid
    return id_to_name, name_to_id

@st.cache_data(show_spinner=False)
def load_games_build_graph_from_text(text: str, id_to_name: Optional[Dict[int, str]] = None):
    """
    Game line format (8 fields):
      day_index, yyyymmdd, team1_id, team1_homeAway(1|-1), team1_score, team2_id, team2_homeAway(1|-1), team2_score
    We ignore home/away and decide winner by score only.
    Nodes are team IDs; edges are winner -> loser with 'games' list.
    """
    G = nx.DiGraph()
    edge_games = defaultdict(list)

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = _split_csvish(line)
        if len(parts) < 8:
            continue
        try:
            day_idx = int(parts[0])
            date = parts[1]
            t1_id = int(parts[2]); t1_score = int(parts[4])
            t2_id = int(parts[5]); t2_score = int(parts[7])
        except ValueError:
            continue

        if t1_score == t2_score:
            # skip ties
            continue

        if t1_score > t2_score:
            winner, loser = t1_id, t2_id
            w_score, l_score = t1_score, t2_score
        else:
            winner, loser = t2_id, t1_id
            w_score, l_score = t2_score, t1_score

        G.add_node(winner)
        G.add_node(loser)
        edge_games[(winner, loser)].append({
            "date": date,
            "day": day_idx,
            "w_id": winner,
            "l_id": loser,
            "w_score": w_score,
            "l_score": l_score,
        })

    for (w, l), games in edge_games.items():
        G.add_edge(w, l, games=games)

    if id_to_name:
        nx.set_node_attributes(G, {tid: {"name": id_to_name.get(tid, str(tid))} for tid in G.nodes})

    return G

# ==========================
# Query utilities
# ==========================

def normalize_team_name(name: str) -> str:
    return name.replace("_", " ").upper().strip()


def shortest_win_chain_ids(G: nx.DiGraph, source_id: int, target_id: int) -> Optional[List[int]]:
    try:
        return nx.shortest_path(G, source=source_id, target=target_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def chain_as_names(path_ids: List[int], id_to_name: Dict[int, str]) -> str:
    return " → ".join(id_to_name.get(t, str(t)) for t in path_ids)


def explain_chain(G: nx.DiGraph, path_ids: List[int], id_to_name: Dict[int, str], pick: str = "earliest") -> List[str]:
    if not path_ids or len(path_ids) < 2:
        return []
    lines = []
    for u, v in zip(path_ids, path_ids[1:]):
        games = G[u][v]["games"]
        if pick == "latest":
            g = max(games, key=lambda x: (x["date"], x["day"]))
        elif pick == "first":
            g = games[0]
        else:
            g = min(games, key=lambda x: (x["date"], x["day"]))
        d = g["date"]
        d_fmt = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        u_name = id_to_name.get(u, str(u))
        v_name = id_to_name.get(v, str(v))
        lines.append(f"{d_fmt}: {u_name} beat {v_name} {g['w_score']}-{g['l_score']}")
    return lines

# ==========================
# Streamlit UI
# ==========================

st.set_page_config(page_title="Win Chain Finder", page_icon="🏀", layout="centered")
st.title("🏀 Shortest Chain of Wins (College Basketball)")
st.write(
    "Upload your **games** and **team names** text files, then pick two teams to compute the shortest winner→loser path."
)

with st.expander("File format details", expanded=False):
    st.markdown(
        """
        **Games file**: each line has 8 comma-separated fields
        
        `day_index, yyyymmdd, team1_id, team1_homeAway(1|-1), team1_score, team2_id, team2_homeAway(1|-1), team2_score`
        
        Home/away flags are ignored; the winner is determined only by the scores.
        
        **Names file**: each line like `10, Arizona` or `1, Abilene_Chr`.
        """
    )

col1, col2 = st.columns(2)
with col1:
    games_file = st.file_uploader("Upload games.txt", type=["txt", "csv"]) 
with col2:
    names_file = st.file_uploader("Upload team_names.txt", type=["txt", "csv"]) 

if games_file and names_file:
    games_text = games_file.read().decode("utf-8", errors="ignore")
    names_text = names_file.read().decode("utf-8", errors="ignore")

    id_to_name, name_to_id = load_team_names_from_text(names_text)
    G = load_games_build_graph_from_text(games_text, id_to_name=id_to_name)

    st.success(f"Loaded graph with {G.number_of_nodes()} teams and {G.number_of_edges()} win-edges.")

    # Team selection
    all_team_names = sorted(id_to_name.values())
    c1, c2 = st.columns(2)
    with c1:
        src_name = st.selectbox("From (winner chain starts at)", all_team_names, index=0)
    with c2:
        dst_name = st.selectbox("To (final team beaten)", all_team_names, index=min(1, len(all_team_names)-1))

    pick_rep = st.radio("Which game to show per step?", ["earliest", "latest", "first"], index=0, horizontal=True)

    if st.button("Compute shortest path"):
        s_id = name_to_id.get(normalize_team_name(src_name))
        t_id = name_to_id.get(normalize_team_name(dst_name))

        if s_id is None or t_id is None:
            st.error("Could not resolve one or both team names.")
        else:
            path_ids = shortest_win_chain_ids(G, s_id, t_id)
            if not path_ids:
                st.warning("No win chain found between those teams in the loaded data.")
            else:
                edges = len(path_ids) - 1
                st.metric("Shortest path length (wins)", edges)
                st.write("**Team chain:**", chain_as_names(path_ids, id_to_name))
                with st.expander("Show game explanations"):
                    for line in explain_chain(G, path_ids, id_to_name, pick=pick_rep):
                        st.write("- ", line)
else:
    st.info("Upload both files to begin.")

st.caption("Tip: If your dataset spans multiple seasons, pre-filter the games text before uploading, or split by date range.")
