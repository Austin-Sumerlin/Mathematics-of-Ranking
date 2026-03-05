# app.py — March Madness R64: Seeds vs Model (Exact Colley + Reseed-by-4)
# -----------------------------------------------------------------------
# - Exact Colley port from user's reference code (no guard/skip logic)
# - Robust Round-of-64 parsing (CURRENT ROUND == 64, pair within BY ROUND NO)
# - Model reseeding: 4 teams per seed (1..16), skipping non-tournament teams
# - Alias + manual name mapping
# - Unique keys for download buttons
# - Head-to-head tab + ESPN-style model bracket view (handles play-ins)

import io
import re
import ast
import json
import math
import uuid
import numpy as np
import pandas as pd
import unicodedata
import streamlit as st
from math import ceil

# --- Make run button persistent so UI does NOT reset on rerun ---
if "run_mode" not in st.session_state:
    st.session_state.run_mode = False

st.set_page_config(page_title="March Madness R64: Seed vs Model", layout="wide")

# =================== Sidebar: Inputs & Controls ===================
st.sidebar.title("Inputs")
teams_file = st.sidebar.file_uploader("Teams file (underscore format)", type=["txt", "csv"], key="teams")
games_file = st.sidebar.file_uploader("Season games file", type=["txt", "csv"], key="games")
bracket_file = st.sidebar.file_uploader("Bracket export (CSV/TSV)", type=["csv", "tsv", "txt"], key="bracket")

st.sidebar.markdown("**avg_spl dictionary** (paste Python dict / CSV, or upload JSON)")
avg_spl_text = st.sidebar.text_area(
    "Paste avg_spl (e.g. {268: 3.48, 30: 3.20, ...} or two columns id,value)", height=120
)
avg_spl_json_file = st.sidebar.file_uploader("...or upload avg_spl.json", type=["json"], key="avgjson")

st.sidebar.header("Model ↔ Seed parameters")
segment_weighting = st.sidebar.text_input(
    "Segment weighting (Python list)", value="[0.5, 1, 2]",
    help="Used for time-weighting within the season",
)
use_weighting = st.sidebar.checkbox("Use time weighting", value=True)

prob_scale = st.sidebar.slider("Logistic scale (prob sharpness)", 1.0, 16.0, 8.0, 0.05)
close_gap_thresh = st.sidebar.slider("Closer-than-seeds threshold (abs gap diff)", 0.0, 8.0, 2.0, 0.5)
blowout_gap_thresh = st.sidebar.slider("More-one-sided threshold (model vs seed gap)", 0.0, 8.0, 2.0, 0.5)
upset_prob_cut = st.sidebar.slider("Upset radar: min win prob for worse seed", 0.0, 0.9, 0.40, 0.01)

st.sidebar.header("Bracket parsing")
tournament_year_input = st.sidebar.text_input(
    "Tournament YEAR to analyze (blank = auto-detect latest)", value=""
)
pair_mode = st.sidebar.selectbox(
    "Matchup pairing mode",
    ["Group by BY ROUND NO (recommended)", "Adjacent rows (fallback)"]
)

st.sidebar.header("Name mapping (optional overrides)")
map_help = (
    "Optional CSV text: 'bracket_name,model_name' per line.\n"
    "Example:\n"
    "Alabama St.,Alabama_St\n"
    "Saint Francis,St_Francis_PA\n"
    "St. John's,St_John's\n"
    "SIU Edwardsville,SIUE\n"
    "UC San Diego,UC_San_Diego\n"
    "Mississippi St.,Mississippi_St\n"
)
user_mapping_text = st.sidebar.text_area("Manual mapping CSV (optional)", help=map_help, height=120)

if st.sidebar.button("Run analysis ➜", type="primary"):
    st.session_state.run_mode = True

st.title("📊 Round of 64 — Compare NCAA Seeds vs Your Model Rankings")
st.caption("Upload files in the sidebar, set YEAR & pairing mode, then click **Run analysis**.")

# =================== Helpers: I/O parsing ===================
def load_teams_df(fbuf) -> pd.DataFrame:
    data = fbuf.read()
    try:
        content = data.decode("utf-8")
    except Exception:
        content = data.decode("latin1")
    raw = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)\s*,\s*(.+?)\s*$", line)
        if m:
            idx = int(m.group(1))
            name = m.group(2)
            raw.append((idx, name))
    if not raw:
        from io import StringIO
        df0 = pd.read_csv(StringIO(content), header=None)
        if df0.shape[1] >= 2:
            df0.columns = ["orig_id_1based", "team_name"] + list(df0.columns[2:])
            df0["team_id"] = df0["orig_id_1based"] - 1
            return df0[["team_id", "team_name"]].sort_values("team_id", ignore_index=True)
        raise ValueError("Teams file not recognized. Expect lines like '1, Abilene_Chr'.")
    df = pd.DataFrame(raw, columns=["orig_id_1based", "team_name"])
    df["team_id"] = df["orig_id_1based"] - 1
    return df[["team_id", "team_name"]].sort_values("team_id", ascending=True, ignore_index=True)

@st.cache_data(show_spinner=False)
def read_games_df(gbuf) -> pd.DataFrame:
    return pd.read_csv(gbuf, header=None)

@st.cache_data(show_spinner=False)
def read_bracket_df(bbuf) -> pd.DataFrame:
    return pd.read_csv(bbuf, sep=r"\s*\t\s*|,", engine="python")

# =================== avg_spl robust parser ===================
def parse_avg_spl(avg_spl_json_file, avg_spl_text):
    if avg_spl_json_file is not None:
        try:
            data = json.load(avg_spl_json_file)
            return {int(k): float(v) for k, v in data.items()}
        except Exception as e:
            st.warning(f"Could not parse avg_spl JSON: {e}")
    txt = (avg_spl_text or "").strip()
    if txt:
        txt_nolead = re.sub(r"^\s*avg_spl\s*=\s*", "", txt)
        m = re.search(r"\{[\s\S]*\}", txt_nolead)
        dict_text = m.group(0) if m else txt_nolead
        try:
            data = ast.literal_eval(dict_text)
            if isinstance(data, dict):
                return {int(k): float(v) for k, v in data.items()}
        except Exception:
            pass
        try:
            rows = []
            for line in txt.splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    rows.append((int(parts[0]), float(parts[1])))
            if rows:
                return dict(rows)
        except Exception:
            pass
    st.info("No valid avg_spl provided — proceeding without SPL boost.")
    return {}

# =================== Colley rankings ===================
def compute_colley_rankings(
    games: pd.DataFrame,
    teams_df: pd.DataFrame,
    avg_spl: dict,
    segmentWeighting=(0.5, 1, 2),
    useWeighting: bool = True,
):
    numTeams = len(teams_df)
    colleyMatrix = 2 * np.diag(np.ones(numTeams))
    b = np.ones(numTeams)

    def weight_avg_spl(x: int) -> float:
        return 3 * math.log10(1 / (avg_spl[x] - 2)) + 1.75

    numGames = len(games)
    dayBeforeSeason = games.loc[0, 0] - 1
    lastDayOfSeason = games.loc[numGames - 1, 0]

    for i in range(numGames):
        team1ID = games.loc[i, 2] - 1
        team1Score = games.loc[i, 4]
        team2ID = games.loc[i, 5] - 1
        team2Score = games.loc[i, 7]
        currentDay = games.loc[i, 0]

        if useWeighting:
            numberSegments = len(segmentWeighting)
            weightIndex = ceil(numberSegments * ((currentDay - dayBeforeSeason) / (lastDayOfSeason - dayBeforeSeason))) - 1
            timeWeight = segmentWeighting[weightIndex]
        else:
            timeWeight = 1

        if team1Score > team2Score:
            if team2ID+1 in avg_spl:
                gameWeight = weight_avg_spl(team2ID+1) * timeWeight
        else:
            if team1ID+1 in avg_spl:
                gameWeight = weight_avg_spl(team1ID+1) * timeWeight

        colleyMatrix[team1ID, team2ID] -= gameWeight
        colleyMatrix[team2ID, team1ID] -= gameWeight
        colleyMatrix[team1ID, team1ID] += gameWeight
        colleyMatrix[team2ID, team2ID] += gameWeight

        if team1Score > team2Score:
            b[team1ID] += 0.5 * gameWeight
            b[team2ID] -= 0.5 * gameWeight
        elif team2Score > team1Score:
            b[team1ID] -= 0.5 * gameWeight
            b[team2ID] += 0.5 * gameWeight

    r = np.linalg.solve(colleyMatrix, b)

    correct = 0
    for i in range(numGames):
        t1 = games.loc[i, 2] - 1
        s1 = games.loc[i, 4]
        t2 = games.loc[i, 5] - 1
        s2 = games.loc[i, 7]
        if (s1 > s2 and r[t1] > r[t2]) or (s2 > s1 and r[t2] > r[t1]):
            correct += 1
    predictability = correct / numGames * 100.0

    iSort = np.argsort(-r)
    ratings = pd.DataFrame({
        "rank": np.arange(1, numTeams + 1, dtype=int),
        "rating": r[iSort],
        "team_name": teams_df.loc[iSort, "team_name"].values,
        "team_id": iSort,
    })
    return ratings, predictability

# =================== Round of 64 parser ===================
def parse_round_of_64(df: pd.DataFrame, *, year: int | None, prefer_grouping: bool = True) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    def need(col: str):
        if col not in cols:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")
    need("team"); need("seed")

    if "year" in cols:
        if year is None:
            latest = int(pd.to_numeric(df[cols["year"]], errors="coerce").dropna().max())
            year = latest
        df = df[pd.to_numeric(df[cols["year"]], errors="coerce") == year].copy()

    round_col = cols.get("round")
    cur_round_col = cols.get("current round")
    if cur_round_col:
        r1 = df[pd.to_numeric(df[cur_round_col], errors="coerce") == 64].copy()
    elif round_col:
        r1 = df[pd.to_numeric(df[round_col], errors="coerce") == 1].copy()
    else:
        r1 = df.copy()

    gcol = cols.get("by round no") if prefer_grouping else None
    tcol = cols["team"]; scol = cols["seed"]

    out_rows = []
    def _emit_pair(gid, a, b):
        teamA = str(a[tcol]).strip()
        teamB = str(b[tcol]).strip()
        out_rows.append({
            "game_id": gid,
            "round": 64,
            "teamA": teamA,
            "seedA": int(pd.to_numeric(a[scol], errors="coerce")),
            "teamB": teamB,
            "seedB": int(pd.to_numeric(b[scol], errors="coerce")),
        })

    if gcol:
        for gid, g in r1.groupby(gcol, sort=False):
            g = g.copy()
            g["_seed"] = pd.to_numeric(g[scol], errors="coerce")
            g = g.sort_values("_seed")
            if len(g) >= 2:
                for j in range(0, len(g)-1, 2):
                    _emit_pair(gid, g.iloc[j], g.iloc[j+1])
    else:
        r1 = r1.reset_index(drop=True)
        for i in range(0, len(r1)-1, 2):
            _emit_pair(i//2, r1.iloc[i], r1.iloc[i+1])

    return pd.DataFrame(out_rows)

# =================== Name mapping ===================
def _ascii_only(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def canon(s: str) -> str:
    s = s.strip().lower()
    s = _ascii_only(s)
    s = s.replace("_", " ")
    s = re.sub(r"[.\-,'’]", " ", s)
    s = re.sub(r"\s+&\s+", " and ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bst\b", "saint", s)
    s = re.sub(r"\bmt\b", "mount", s)
    return s

DEFAULT_ALIAS = {
    "north carolina st": "NC_State",
    "north carolina state": "NC_State",
    "nc state": "NC_State",
    "st john s": "St_John's",
    "st francis": "St_Francis_PA",
    "mount st mary s": "Mt_St_Mary's",
    "western kentucky": "WKU",
    "alabama st": "Alabama_St",
    "siu edwardsville": "SIUE",
    "texas a m": "Texas_A&M",
    "mississippi st": "Mississippi_St",
}

def build_team_index(teams_df: pd.DataFrame) -> dict:
    idx = {}
    for _, r in teams_df.iterrows():
        c = canon(str(r["team_name"]).replace("_", " "))
        idx[c] = str(r["team_name"])
    return idx

def token_score(a: str, b: str) -> float:
    sa = set(a.split()); sb = set(b.split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

def parse_user_mapping_csv(text: str) -> dict:
    if not text.strip():
        return {}
    mp = {}
    for line in text.splitlines():
        if "," in line:
            left, right = line.split(",", 1)
            mp[left.strip()] = right.strip()
    return mp

def map_names(teams_df, names, user_map_text):
    idx = build_team_index(teams_df)
    alias = DEFAULT_ALIAS.copy()
    user_map = parse_user_mapping_csv(user_map_text)

    mapped = {}
    for name in names.unique():
        raw = str(name)
        c = canon(raw)

        if raw in user_map:
            mapped[raw] = user_map[raw]
            continue

        c_alias_raw = alias.get(c, c)
        c_alias_key = canon(c_alias_raw)

        if c_alias_key in idx:
            mapped[raw] = idx[c_alias_key]
            continue

        if c in idx:
            mapped[raw] = idx[c]
            continue

        best = None; best_score = -1
        for c_key, team in idx.items():
            sc = token_score(c_alias_key, c_key)
            if sc > best_score:
                best_score = sc
                best = team
        mapped[raw] = best if best_score >= 0.5 else None

    return mapped

# =================== Model reseeding ===================
def build_model_reseed_mapping(tournament_team_set, rankings):
    mapping = {}
    seed = 1
    bucket = 0
    for _, row in rankings.sort_values("rank").iterrows():
        t = row["team_name"]
        if t not in tournament_team_set:
            continue
        mapping[t] = seed
        bucket += 1
        if bucket == 4:
            seed += 1
            bucket = 0
        if seed > 16:
            break
    return mapping

# =================== Probabilities & annotation ===================
def implied_seed_from_rank(rank, n_teams):
    bin_size = n_teams / 16
    return int(min(max(math.ceil(rank / bin_size), 1), 16))

def logistic_prob(rA, rB, scale=8.0):
    return 1 / (1 + math.exp(-(rA - rB) * scale))

def join_and_annotate(
    games64, rankings, teams_df,
    prob_scale, close_gap, blowout_gap, upset_cut, user_map_text
):
    if games64.empty:
        return pd.DataFrame(), []

    name_map = map_names(teams_df, pd.concat([games64["teamA"], games64["teamB"]]), user_map_text)
    out = games64.copy()

    out["teamA_model"] = out["teamA"].map(name_map)
    out["teamB_model"] = out["teamB"].map(name_map)

    model = rankings[["team_name", "rating", "rank"]]

    out = (
        out.merge(model, left_on="teamA_model", right_on="team_name", how="left")
          .rename(columns={"rating": "ratingA", "rank": "rankA"})
          .drop(columns=["team_name"])
          .merge(model, left_on="teamB_model", right_on="team_name", how="left")
          .rename(columns={"rating": "ratingB", "rank": "rankB"})
          .drop(columns=["team_name"])
    )

    resolved = set(out["teamA_model"].dropna()) | set(out["teamB_model"].dropna())
    reseed = build_model_reseed_mapping(resolved, rankings)

    out["modelReseedA"] = out["teamA_model"].map(reseed)
    out["modelReseedB"] = out["teamB_model"].map(reseed)

    n_teams = len(rankings)
    out["impliedSeedA"] = out["rankA"].apply(lambda r: implied_seed_from_rank(r, n_teams) if pd.notna(r) else None)
    out["impliedSeedB"] = out["rankB"].apply(lambda r: implied_seed_from_rank(r, n_teams) if pd.notna(r) else None)

    out["pA"] = out.apply(
        lambda r: logistic_prob(r["ratingA"], r["ratingB"], prob_scale)
        if pd.notna(r["ratingA"]) and pd.notna(r["ratingB"]) else float("nan"),
        axis=1
    )
    out["pB"] = 1 - out["pA"]

    out["seedGap"] = out["seedB"] - out["seedA"]
    out["modelSeedGap"] = out["modelReseedB"] - out["modelReseedA"]
    out["gap_discrepancy"] = (out["seedGap"] - out["modelSeedGap"]).abs()

    out["closer_than_seeds"] = (out["seedGap"].abs() >= close_gap) & (
        out["modelSeedGap"].abs() < out["seedGap"].abs()
    )
    out["more_one_sided"] = (out["modelSeedGap"].abs() - out["seedGap"].abs() >= blowout_gap)

    worse_A = out["seedA"] > out["seedB"]
    out["upset_radar"] = (
        (worse_A & (out["pA"] >= upset_cut))
        | (~worse_A & (out["pB"] >= upset_cut))
    )

    unresolved = sorted([k for k, v in name_map.items() if v is None])
    return out, unresolved

# -------------------------------------------------------------------
# ========== NEW HELPERS FOR ESPN-STYLE BRACKET VIEW =================
# -------------------------------------------------------------------

ROUND1_SEED_ORDER = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]

def safe_logistic(rA, rB, scale):
    """Guarded logistic for bracket view (returns None if any rating missing)."""
    if rA is None or rB is None:
        return None
    try:
        if math.isnan(rA) or math.isnan(rB):
            return None
    except TypeError:
        pass
    return logistic_prob(rA, rB, scale)

def build_region_team_table(bracket_df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    """
    Correct region splitter:
    A region ends ONLY when we see *seed=15 AND previous row seed=2*.
    This matches the 2/15 matchup exactly and avoids false 15s.
    """

    cols = {c.lower().strip(): c for c in bracket_df.columns}
    if "team" not in cols or "seed" not in cols:
        return pd.DataFrame(columns=["team","seed","region"])

    df = bracket_df.copy()

    # Filter by year
    if "year" in cols and year is not None:
        df = df[pd.to_numeric(df[cols["year"]], errors="coerce") == year]

    df = df[[cols["team"], cols["seed"]]].copy()
    df.columns = ["team", "seed"]

    df["team"] = df["team"].astype(str).str.strip()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df[df["team"] != ""].reset_index(drop=True)

    regions = []
    buffer = []

    prev_seed = None

    for _, row in df.iterrows():

        buffer.append(row)

        # Detect ACTUAL region-ending 2/15 matchup
        if row["seed"] == 15 and prev_seed == 2:
            # finalize this region
            reg = pd.DataFrame(buffer)
            reg["region"] = f"Region {len(regions)+1}"
            regions.append(reg)
            buffer = []  # start next region

        prev_seed = row["seed"]

        if len(regions) == 4:
            break

    if not regions:
        return pd.DataFrame(columns=["team","seed","region"])

    return pd.concat(regions, ignore_index=True)

def build_region_slots(region_df: pd.DataFrame, teams_df: pd.DataFrame,
                       rankings_df: pd.DataFrame, user_map_text: str):
    """
    Build mapping: seed -> list of team dicts for a single region.
    Each team dict has bracket_name, model_name, rating, rank, seed.
    """
    if region_df.empty:
        return {}

    name_map = map_names(teams_df, region_df["team"], user_map_text)
    ratings_lookup = rankings_df.set_index("team_name")[["rating", "rank"]].to_dict(orient="index")

    slots = {}
    for _, row in region_df.iterrows():
        seed = int(row["seed"])
        bracket_name = row["team"]
        model_name = name_map.get(bracket_name)
        rating = None
        rank = None
        if model_name in ratings_lookup:
            rating = float(ratings_lookup[model_name]["rating"])
            rank = int(ratings_lookup[model_name]["rank"])

        team_obj = {
            "bracket_name": bracket_name,
            "model_name": model_name,
            "rating": rating,
            "rank": rank,
            "seed": seed,
        }
        slots.setdefault(seed, []).append(team_obj)

    return slots

def choose_playin_winner(team_list):
    """Winner rule for play-ins: higher model rating wins (ties -> first)."""
    valid = [t for t in team_list if t["rating"] is not None]
    if not valid:
        return team_list[0]
    return max(valid, key=lambda t: t["rating"])

def simulate_region_bracket(slots, prob_scale):
    """
    Simulate a single region: returns dict with rounds:
      round1: list of games with play-in info and per-opponent probs
      round2..4: standard games between single teams
    """
    # Round of 64
    round1_games = []
    winners_r1 = []

    for seed_hi, seed_lo in ROUND1_SEED_ORDER:
        if seed_hi not in slots or seed_lo not in slots:
            continue

        slot_hi = slots[seed_hi]
        slot_lo = slots[seed_lo]

        team_hi = choose_playin_winner(slot_hi)
        team_lo = choose_playin_winner(slot_lo)

        p_main = safe_logistic(team_hi["rating"], team_lo["rating"], prob_scale)

        pA_vs_each_B = []
        pB_vs_each_B = []
        for opp in slot_lo:
            p = safe_logistic(team_hi["rating"], opp["rating"], prob_scale)
            if p is None:
                pA_vs_each_B.append(None)
                pB_vs_each_B.append(None)
            else:
                pA_vs_each_B.append(p)
                pB_vs_each_B.append(1 - p)

        if p_main is None or p_main >= 0.5:
            winner = team_hi
        else:
            winner = team_lo

        game = {
            "round": 1,
            "seedA": seed_hi,
            "seedB": seed_lo,
            "teamA": team_hi,
            "teamB": team_lo,
            "slotA": slot_hi,
            "slotB": slot_lo,
            "pA_main": p_main,
            "pB_main": None if p_main is None else 1 - p_main,
            "pA_vs_each_B": pA_vs_each_B,
            "pB_vs_each_B": pB_vs_each_B,
            "winner": winner,
        }
        round1_games.append(game)
        winners_r1.append(winner)

    # Helper for later rounds (no play-ins)
    def next_round(winners, round_num):
        games = []
        next_winners = []
        for i in range(0, len(winners), 2):
            if i + 1 >= len(winners):
                break
            tA = winners[i]
            tB = winners[i+1]
            pA = safe_logistic(tA["rating"], tB["rating"], prob_scale)
            if pA is None or pA >= 0.5:
                winner = tA
            else:
                winner = tB
            game = {
                "round": round_num,
                "teamA": tA,
                "teamB": tB,
                "pA": pA,
                "pB": None if pA is None else 1 - pA,
                "winner": winner,
            }
            games.append(game)
            next_winners.append(winner)
        return games, next_winners

    round2_games, winners_r2 = next_round(winners_r1, 2)
    round3_games, winners_r3 = next_round(winners_r2, 3)
    round4_games, winners_r4 = next_round(winners_r3, 4)  # regional final (len 0 or 1)

    return {
        "round1": round1_games,
        "round2": round2_games,
        "round3": round3_games,
        "round4": round4_games,
        "champion": winners_r4[0] if winners_r4 else None,
    }

def simulate_final_four(regional_champions, prob_scale):
    """
    Input: list of exactly 4 team dicts (region champions)
    Output: dict with:
        - semifinal1 (Region1 vs Region2)
        - semifinal2 (Region3 vs Region4)
        - final
        - champion
    """

    if len(regional_champions) != 4:
        return None

    A, B, C, D = regional_champions  # fixed bracket order

    # --- Semifinal 1 ---
    pA = safe_logistic(A["rating"], B["rating"], prob_scale)
    winner1 = A if (pA is None or pA >= 0.5) else B

    semi1 = {
        "teamA": A, "teamB": B,
        "pA": pA, "pB": None if pA is None else 1 - pA,
        "winner": winner1
    }

    # --- Semifinal 2 ---
    pC = safe_logistic(C["rating"], D["rating"], prob_scale)
    winner2 = C if (pC is None or pC >= 0.5) else D

    semi2 = {
        "teamA": C, "teamB": D,
        "pA": pC, "pB": None if pC is None else 1 - pC,
        "winner": winner2
    }

    # --- Championship ---
    t1 = winner1
    t2 = winner2
    p_final = safe_logistic(t1["rating"], t2["rating"], prob_scale)
    champ = t1 if (p_final is None or p_final >= 0.5) else t2

    final_game = {
        "teamA": t1, "teamB": t2,
        "pA": p_final, "pB": None if p_final is None else 1 - p_final,
        "winner": champ,
    }

    return {
        "semifinal1": semi1,
        "semifinal2": semi2,
        "final": final_game,
        "champion": champ,
    }

def format_prob_list(probs):
    vals = []
    for p in probs:
        if p is None:
            vals.append("?")
        else:
            vals.append(f"{p*100:.1f}%")
    return " / ".join(vals) if vals else "—"

def render_region_bracket(region_label, region_slots, region_sim):
    """Render one region in ESPN-style columns."""
    st.markdown(f"### {region_label}")

    col_r64, col_r32, col_s16, col_e8 = st.columns(4)

    # Round of 64 (with play-in info)
    with col_r64:
        st.markdown("**Round of 64**")
        if not region_sim["round1"]:
            st.info("No Round-of-64 games for this region.")
        for g in region_sim["round1"]:
            # High seed side: single team
            tA = g["teamA"]
            nameA = tA["bracket_name"]
            seedA = g["seedA"]

            # Play-in / low-seed side
            slotB = g["slotB"]
            namesB = " / ".join(t["bracket_name"] for t in slotB)
            seedB = g["seedB"]

            pA_str = format_prob_list(g["pA_vs_each_B"])
            pB_str = format_prob_list(g["pB_vs_each_B"])

            html = f"""
            <div style="border-radius:6px; padding:6px 8px; margin-bottom:8px;
                        background-color:#f7f7f7; border:1px solid #ddd; font-size:0.85rem;">
              <div>
                <strong>{seedA} {nameA}</strong>
                <span style="float:right;">{pA_str}</span>
              </div>
              <div style="clear:both;"></div>
              <div>
                <strong>{seedB} {namesB}</strong>
                <span style="float:right;">{pB_str}</span>
              </div>
              <div style="clear:both;"></div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    # Helper for later rounds
    def render_simple_round(col, title, games):
        with col:
            st.markdown(f"**{title}**")
            if not games:
                st.info("No games.")
                return
            for g in games:
                tA = g["teamA"]
                tB = g["teamB"]
                seedA = tA["seed"]
                seedB = tB["seed"]
                pA = g["pA"]
                pB = g["pB"]
                pA_str = "?" if pA is None else f"{pA*100:.1f}%"
                pB_str = "?" if pB is None else f"{pB*100:.1f}%"
                html = f"""
                <div style="border-radius:6px; padding:6px 8px; margin-bottom:8px;
                            background-color:#f7f7f7; border:1px solid #ddd; font-size:0.85rem;">
                  <div>
                    <strong>{seedA} {tA['bracket_name']}</strong>
                    <span style="float:right;">{pA_str}</span>
                  </div>
                  <div style="clear:both;"></div>
                  <div>
                    <strong>{seedB} {tB['bracket_name']}</strong>
                    <span style="float:right;">{pB_str}</span>
                  </div>
                  <div style="clear:both;"></div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

    render_simple_round(col_r32, "Round of 32", region_sim["round2"])
    render_simple_round(col_s16, "Sweet 16", region_sim["round3"])
    render_simple_round(col_e8, "Elite 8 (Regional Final)", region_sim["round4"])

    champ = region_sim["champion"]
    if champ is not None:
        st.markdown(
            f"<div style='margin-top:8px; padding:6px 8px; border-radius:6px;"
            f"background-color:#ffeeba; border:1px solid #f0ad4e; font-size:0.85rem;'>"
            f"<strong>Regional champion (model):</strong> "
            f"{champ['seed']} {champ['bracket_name']}"
            f"</div>",
            unsafe_allow_html=True,
        )

def render_final_four(ff):
    st.markdown("## Final Four")

    def display_game(title, g):
        tA, tB = g["teamA"], g["teamB"]
        pA, pB = g["pA"], g["pB"]
        pA_str = "?" if pA is None else f"{pA*100:.1f}%"
        pB_str = "?" if pB is None else f"{pB*100:.1f}%"

        st.markdown(f"### {title}")
        html = f"""
        <div style="border-radius:6px; padding:8px; margin-bottom:10px;
                    background-color:#f7f7f7; border:1px solid #ddd;">
          <div><strong>{tA['bracket_name']}</strong> <span style="float:right;">{pA_str}</span></div>
          <div style="clear:both;"></div>
          <div><strong>{tB['bracket_name']}</strong> <span style="float:right;">{pB_str}</span></div>
          <div style="clear:both;"></div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    display_game("Semifinal 1 (Region 1 vs Region 2)", ff["semifinal1"])
    display_game("Semifinal 2 (Region 3 vs Region 4)", ff["semifinal2"])

    st.markdown("## National Championship")
    display_game("Championship Game", ff["final"])

    champ = ff["champion"]
    st.markdown(
        f"<div style='margin-top:12px; padding:10px; border-radius:6px;"
        f"background-color:#ffeeba; border:1px solid #f0ad4e; font-size:1rem;'>"
        f"<strong> National Champion (Model):</strong> "
        f"{champ['bracket_name']}"
        f"</div>",
        unsafe_allow_html=True,
    )

# =================== UI ===================
if st.session_state.run_mode:
    if not (teams_file and games_file and bracket_file):
        st.error("Please upload Teams, Games, and Bracket files.")
        st.stop()

    avg_spl = parse_avg_spl(avg_spl_json_file, avg_spl_text)

    try:
        segw = ast.literal_eval(segment_weighting)
    except:
        segw = [0.5, 1, 2]
    if not isinstance(segw, (list, tuple)):
        segw = [0.5, 1, 2]

    teams_df = load_teams_df(teams_file)
    games_df = read_games_df(games_file)
    bracket_df = read_bracket_df(bracket_file)

    chosen_year = None
    if tournament_year_input.strip():
        try:
            chosen_year = int(tournament_year_input.strip())
        except:
            pass

    rankings_df, predictability = compute_colley_rankings(
        games_df, teams_df, avg_spl,
        segmentWeighting=tuple(segw), useWeighting=use_weighting
    )

    games64 = parse_round_of_64(
        bracket_df,
        year=chosen_year,
        prefer_grouping=(pair_mode.startswith("Group by"))
    )

    tmp_map = map_names(teams_df, pd.concat([games64["teamA"], games64["teamB"]]), user_mapping_text)
    unresolved_pre = sorted([k for k, v in tmp_map.items() if k is not None and v is None])
    if unresolved_pre:
        with st.expander("⚠️ Unresolved bracket names — add mapping overrides", expanded=True):
            st.write(pd.DataFrame({"bracket_name": unresolved_pre}))

    annotated, unresolved = join_and_annotate(
        games64, rankings_df, teams_df,
        prob_scale, close_gap_thresh, blowout_gap_thresh, upset_prob_cut,
        user_mapping_text
    )

    st.subheader("Season summary")
    colA, colB, colC = st.columns(3)
    colA.metric("Teams", len(teams_df))
    colB.metric("Games", len(games_df))
    colC.metric("Predictability", f"{predictability:.2f}%")

    if unresolved:
        with st.expander("⚠️ Remaining unresolved names", expanded=False):
            st.write(pd.DataFrame({"bracket_name": unresolved}))

    # ==========================
    #     TABS (incl. new one)
    # ==========================
    tabs = st.tabs([
        "All Round of 64",
        "Upset radar",
        "Closer than seeds",
        "More one-sided",
        "Rankings (model)",
        "Head-to-Head",
        "Model Bracket (ESPN-style)",
    ])

    base_cols = [
        "game_id",
        "teamA","seedA","teamA_model","rankA","modelReseedA","ratingA","pA",
        "teamB","seedB","teamB_model","rankB","modelReseedB","ratingB","pB",
        "seedGap","modelSeedGap","gap_discrepancy",
        "closer_than_seeds","more_one_sided","upset_radar"
    ]

    def _show(df, container, key_suffix):
        if df.empty:
            container.info("No games match this filter.")
            return
        view = df[base_cols].sort_values(
            ["upset_radar", "gap_discrepancy"], ascending=[False, False]
        )
        container.dataframe(view, use_container_width=True)
        csv_buf = io.StringIO(); view.to_csv(csv_buf, index=False)
        container.download_button(
            "Download CSV", csv_buf.getvalue(),
            f"round64_{key_suffix}.csv", mime="text/csv",
            key=f"dl_{key_suffix}_{uuid.uuid4().hex}"
        )

    # Existing tabs
    with tabs[0]:
        _show(annotated, st, "all")
    with tabs[1]:
        _show(annotated[annotated["upset_radar"]], st, "upset")
    with tabs[2]:
        _show(annotated[annotated["closer_than_seeds"]], st, "closer")
    with tabs[3]:
        _show(annotated[annotated["more_one_sided"]], st, "one_sided")
    with tabs[4]:
        st.dataframe(rankings_df.sort_values("rank"), use_container_width=True)

    # ======================================================
    # =============  HEAD-TO-HEAD TAB  =====================
    # ======================================================
    with tabs[5]:
        st.subheader("Head-to-Head Win Probability")

        team_options = rankings_df["team_name"].sort_values().tolist()

        c1, c2 = st.columns(2)
        with c1:
            tA = st.selectbox("Team A", team_options, key="h2hA")
        with c2:
            tB = st.selectbox("Team B", team_options, key="h2hB")

        if tA and tB:
            rowA = rankings_df[rankings_df["team_name"] == tA].iloc[0]
            rowB = rankings_df[rankings_df["team_name"] == tB].iloc[0]

            pA = logistic_prob(rowA["rating"], rowB["rating"], scale=prob_scale)
            pB = 1 - pA

            st.metric(f"{tA} win probability", f"{pA*100:.1f}%")
            st.metric(f"{tB} win probability", f"{pB*100:.1f}%")

            st.write(f"**Ratings:** {tA} = {rowA['rating']:.4f}, {tB} = {rowB['rating']:.4f}")
            st.write(f"**Ranks:** {tA} = #{rowA['rank']}, {tB} = #{rowB['rank']}")

    # ======================================================
    # =============  MODEL BRACKET TAB  ====================
    # ======================================================
    with tabs[6]:
        st.subheader("Model Bracket — ESPN-style View (with Play-ins)")

        region_team_table = build_region_team_table(bracket_df, chosen_year)

        if region_team_table.empty:
            st.info("Could not build region/team table from bracket file.")
        else:
            regional_champs = []

            for region_label in sorted(region_team_table["region"].unique()):
                region_df = region_team_table[region_team_table["region"] == region_label]

                slots = build_region_slots(region_df, teams_df, rankings_df, user_mapping_text)
                if not slots:
                    st.info(f"No data for {region_label}.")
                    continue

                sim = simulate_region_bracket(slots, prob_scale)
                render_region_bracket(region_label, slots, sim)

                # Save region winners for Final Four
                if sim["champion"] is not None:
                    regional_champs.append(sim["champion"])

            # ======== FINAL FOUR & NATIONAL CHAMPION =========
            if len(regional_champs) == 4:
                ff = simulate_final_four(regional_champs, prob_scale)
                render_final_four(ff)
            else:
                st.warning("Not all regional champions were computed; cannot build Final Four.")


else:
    st.info("⬅️ Upload files, set YEAR & pairing mode, then click Run analysis.")
