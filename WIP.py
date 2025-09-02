# pip install networkx
import re
import networkx as nx
from collections import defaultdict

# ------------ Parsing helpers ------------

def _split_csvish(line: str):
    """Split 'csv with loose spaces' lines."""
    return re.split(r"\s*,\s*", line.strip())

def load_team_names(names_txt_path):
    """
    Lines like: 10, Arizona   |   1, Abilene_Chr
    Returns:
      id_to_name: {10: 'Arizona', 1: 'Abilene Chr'}
      name_to_id: {'ARIZONA': 10, 'ABILENE CHR': 1}
    """
    id_to_name, name_to_id = {}, {}
    with open(names_txt_path, encoding="utf-8") as f:
        for raw in f:
            if not raw.strip(): 
                continue
            parts = _split_csvish(raw)
            if len(parts) < 2:
                continue
            try:
                tid = int(parts[0])
            except ValueError:
                continue
            # Normalize underscores -> spaces for display, uppercase key for lookup
            name = re.sub(r"\s+", " ", parts[1].replace("_", " ")).strip()
            id_to_name[tid] = name
            name_to_id[name.upper()] = tid
    return id_to_name, name_to_id

def load_games_build_graph(games_txt_path, id_to_name=None):
    """
    Game line format (8 fields):
      day_index, yyyymmdd, team1_id, team1_homeAway(1|-1), team1_score, team2_id, team2_homeAway(1|-1), team2_score
    We ignore home/away and decide winner by score only.
    Graph nodes are team IDs; edges are winner -> loser with 'games' metadata list.
    """
    G = nx.DiGraph()
    edge_games = defaultdict(list)

    with open(games_txt_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = _split_csvish(line)
            if len(parts) < 8:
                continue  # malformed row
            try:
                day_idx = int(parts[0])
                date = parts[1]
                t1_id = int(parts[2]); t1_score = int(parts[4])
                t2_id = int(parts[5]); t2_score = int(parts[7])
            except ValueError:
                continue

            if t1_score == t2_score:
                # ties: skip or handle per your rules
                continue

            if t1_score > t2_score:
                winner, loser = t1_id, t2_id
                w_score, l_score = t1_score, t2_score
            else:
                winner, loser = t2_id, t1_id
                w_score, l_score = t2_score, t1_score

            # Ensure nodes exist
            G.add_node(winner)
            G.add_node(loser)

            # Accumulate all game instances on this directed edge
            edge_games[(winner, loser)].append({
                "date": date,
                "day": day_idx,
                "w_id": winner, "l_id": loser,
                "w_score": w_score, "l_score": l_score,
            })

    # Attach games to edges
    for (w, l), games in edge_games.items():
        G.add_edge(w, l, games=games)

    # Optional: stash pretty names
    if id_to_name:
        nx.set_node_attributes(G, {tid: {"name": id_to_name.get(tid, str(tid))} for tid in G.nodes})
    return G

# ------------ Shortest win chain ------------

def shortest_win_chain_ids(G, source_id, target_id):
    """Shortest chain of wins by team IDs."""
    try:
        return nx.shortest_path(G, source=source_id, target=target_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def shortest_win_chain_names(G, name_to_id, source_name, target_name):
    """Wrapper to query by names (underscores or spaces OK)."""
    s = source_name.replace("_", " ").upper()
    t = target_name.replace("_", " ").upper()
    s_id = name_to_id.get(s); t_id = name_to_id.get(t)
    if s_id is None or t_id is None:
        return None
    return shortest_win_chain_ids(G, s_id, t_id)

# ------------ Pretty output ------------

def team_label(tid, id_to_name):
    return id_to_name.get(tid, str(tid))

def chain_as_names(path_ids, id_to_name):
    if not path_ids: return "No path"
    return " → ".join(team_label(t, id_to_name) for t in path_ids)

def explain_chain(G, path_ids, id_to_name=None, pick="earliest"):
    """
    For each edge in the path, print one representative game.
    pick: 'first' | 'earliest' | 'latest'
    """
    if not path_ids or len(path_ids) < 2:
        return []
    lines = []
    for u, v in zip(path_ids, path_ids[1:]):
        games = G[u][v]["games"]
        if pick == "earliest":
            g = min(games, key=lambda x: (x["date"], x["day"]))
        elif pick == "latest":
            g = max(games, key=lambda x: (x["date"], x["day"]))
        else:
            g = games[0]
        d = g["date"]
        date_fmt = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        u_name = team_label(u, id_to_name or {})
        v_name = team_label(v, id_to_name or {})
        lines.append(f"{date_fmt}: {u_name} beat {v_name} {g['w_score']}-{g['l_score']}")
    return lines

# ------------ Example usage ------------
names_path = "names.txt"
games_path = "games.txt"

id_to_name, name_to_id = load_team_names(names_path)
G = load_games_build_graph(games_path, id_to_name=id_to_name)

path_ids = shortest_win_chain_names(G, name_to_id, "Arizona", "Abilene_Chr")
if path_ids:
    print(chain_as_names(path_ids, id_to_name))
    for line in explain_chain(G, path_ids, id_to_name, pick="earliest"):
        print(line)
else:
    print("No win chain found.")
