# app.py — March Madness R64: Seeds vs Model (Exact Colley + Reseed-by-4)
# -----------------------------------------------------------------------
# - Exact Colley port from user's reference code (no guard/skip logic)
# - Robust Round-of-64 parsing (CURRENT ROUND == 64, pair within BY ROUND NO)
# - Model reseeding: 4 teams per seed (1..16), skipping non-tournament teams
# - Alias + manual name mapping
# - Unique keys for download buttons

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

run_btn = st.sidebar.button("Run analysis ➜", type="primary")

st.title("📊 Round of 64 — Compare NCAA Seeds vs Your Model Rankings")
st.caption("Upload files in the sidebar, set YEAR & pairing mode, then click **Run analysis**.")

# =================== Helpers: I/O parsing ===================
def load_teams_df(fbuf) -> pd.DataFrame:
    """Reads teams file like '  1, Abilene_Chr' into [team_id (0-based), team_name]."""
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
    # Support csv or tsv
    return pd.read_csv(bbuf, sep=r"\s*\t\s*|,", engine="python")

# =================== avg_spl robust parser ===================
def parse_avg_spl(avg_spl_json_file, avg_spl_text):
    # 1) JSON upload
    if avg_spl_json_file is not None:
        try:
            data = json.load(avg_spl_json_file)
            return {int(k): float(v) for k, v in data.items()}
        except Exception as e:
            st.warning(f"Could not parse avg_spl JSON: {e}")
    # 2) Text: allow 'avg_spl = {...}', or raw dict, or 2-col CSV
    txt = (avg_spl_text or "").strip()
    if txt:
        txt_nolead = re.sub(r"^\s*avg_spl\s*=\s*", "", txt)
        m = re.search(r"\{[\s\S]*\}", txt_nolead)
        dict_text = m.group(0) if m else txt_nolead
        try:
            data = ast.literal_eval(dict_text)
            if isinstance(data, dict):
                return {int(k): float(v) for k, v in data.items()}
        except Exception as e:
            st.warning(f"Could not parse avg_spl text as a Python dict: {e}")
        # CSV fallback: key,value per line
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

# =================== Colley rankings (EXACT reference behavior) ===================
def compute_colley_rankings(
    games: pd.DataFrame,
    teams_df: pd.DataFrame,
    avg_spl: dict,
    segmentWeighting=(0.5, 1, 2),
    useWeighting: bool = True,
):
    """
    Exact port of the user's reference script:
      - timeWeight index uses ceil(...)-1 (no clipping)
      - NO guarding/skip if an avg_spl key is missing (will raise KeyError, same as reference)
      - gameWeight = weight_avg_spl(loser_id) * timeWeight
      - identical Colley updates and 0.5 win/loss handling
      - predictability identical
    """
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

        # Time weight (identical indexing math)
        if useWeighting:
            numberSegments = len(segmentWeighting)
            weightIndex = ceil(
                numberSegments * ((currentDay - dayBeforeSeason) / (lastDayOfSeason - dayBeforeSeason))
            ) - 1
            timeWeight = segmentWeighting[weightIndex]
        else:
            timeWeight = 1

        # EXACT winner branch: weight by LOSER's avg_spl value (no guards)
        if team1Score > team2Score:      # Team 1 won
            if team2ID+1 in avg_spl:
                gameWeight = weight_avg_spl(team2ID+1) * timeWeight
        else:                             # Team 2 won (or tie not expected in CBB)
            if team1ID+1 in avg_spl:
                gameWeight = weight_avg_spl(team1ID+1) * timeWeight

        # Colley matrix updates
        colleyMatrix[team1ID, team2ID] -= gameWeight
        colleyMatrix[team2ID, team1ID] -= gameWeight
        colleyMatrix[team1ID, team1ID] += gameWeight
        colleyMatrix[team2ID, team2ID] += gameWeight

        # RHS updates (0.5)
        if team1Score > team2Score:
            b[team1ID] += 0.5 * gameWeight
            b[team2ID] -= 0.5 * gameWeight
        elif team2Score > team1Score:
            b[team1ID] -= 0.5 * gameWeight
            b[team2ID] += 0.5 * gameWeight
        else:
            # tie case in the reference added 0; we mirror that (no-op)
            b[team1ID] += 0
            b[team2ID] += 0

    r = np.linalg.solve(colleyMatrix, b)

    # Predictability (identical)
    correct = 0
    for i in range(numGames):
        t1 = games.loc[i, 2] - 1
        s1 = games.loc[i, 4]
        t2 = games.loc[i, 5] - 1
        s2 = games.loc[i, 7]
        if (s1 > s2 and r[t1] > r[t2]) or (s2 > s1 and r[t2] > r[t1]) or (s1 == s2 and r[t1] == r[t2]):
            correct += 1
    predictability = correct / numGames * 100.0

    # Rankings in print order
    iSort = np.argsort(-r)
    ratings = pd.DataFrame({
        "rank": np.arange(1, numTeams + 1, dtype=int),
        "rating": r[iSort],
        "team_name": teams_df.loc[iSort, "team_name"].values,
        "team_id": iSort,
    })
    return ratings, predictability

# =================== Round of 64 parser (strict) ===================
def parse_round_of_64(df: pd.DataFrame, *, year: int | None, prefer_grouping: bool = True) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    def need(col: str) -> None:
        if col not in cols:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")
    need("team"); need("seed")

    # YEAR filter
    if "year" in cols:
        if year is None:
            latest = int(pd.to_numeric(df[cols["year"]], errors="coerce").dropna().max())
            year = latest
        df = df[pd.to_numeric(df[cols["year"]], errors="coerce") == year].copy()
        if df.empty:
            st.error(f"No rows for YEAR={year}.")
            return pd.DataFrame(columns=["game_id","round","teamA","seedA","teamB","seedB"])

    round_col = cols.get("round")
    cur_round_col = cols.get("current round")
    if cur_round_col:
        r1 = df[pd.to_numeric(df[cur_round_col], errors="coerce") == 64].copy()
    elif round_col:
        r1 = df[pd.to_numeric(df[round_col], errors="coerce") == 1].copy()
        if "current round" in cols:
            r1 = r1[pd.to_numeric(r1[cols["current round"]], errors="coerce") == 64]
    else:
        st.warning("No CURRENT ROUND/ROUND column found; using entire file (may include later rounds).")
        r1 = df.copy()

    if r1.empty:
        st.warning("No rows matched Round-of-64 markers for the selected YEAR.")
        return pd.DataFrame(columns=["game_id","round","teamA","seedA","teamB","seedB"])

    gcol = cols.get("by round no") if prefer_grouping else None
    tcol = cols["team"]; scol = cols["seed"]
    out_rows = []

    def _emit_pair(gid, a, b):
        teamA = str(a[tcol]).strip(); teamB = str(b[tcol]).strip()
        if teamA == teamB:
            return
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
            g = g.sort_values(["_seed"], kind="stable").drop(columns=["_seed"])
            if len(g) == 2:
                a, b = g.iloc[0], g.iloc[1]
                _emit_pair(gid, a, b)
            elif len(g) == 4:
                seeds = pd.to_numeric(g[scol], errors="coerce").tolist()
                min_seed = min(seeds); max_seed = max(seeds)
                hi = g[g[scol].astype(str) == str(max_seed)]
                lo = g[g[scol].astype(str) == str(min_seed)]
                if len(hi) == 2 and len(lo) == 2:
                    lo_iter = list(lo.itertuples(index=False))
                    hi_iter = list(hi.itertuples(index=False))
                    for a_tup, b_tup in zip(lo_iter, hi_iter):
                        _emit_pair(gid, pd.Series(a_tup._asdict()), pd.Series(b_tup._asdict()))
                else:
                    for j in range(0, len(g) - 1, 2):
                        _emit_pair(gid, g.iloc[j], g.iloc[j+1])
            else:
                for j in range(0, len(g) - 1, 2):
                    _emit_pair(gid, g.iloc[j], g.iloc[j+1])
    else:
        r1 = r1.reset_index(drop=True)
        for i in range(0, len(r1) - 1, 2):
            _emit_pair(i // 2, r1.iloc[i], r1.iloc[i+1])

    out = pd.DataFrame(out_rows, columns=["game_id","round","teamA","seedA","teamB","seedB"])
    if out.empty:
        st.warning("No Round-of-64 pairs detected after YEAR filter and grouping logic.")
    return out

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
    # State / "St." disambiguations
    "north carolina st": "NC_State",
    "north carolina state": "NC_State",
    "n c state": "NC_State",
    "nc state": "NC_State",

    # Saint / St. schools (common variants)
    "st john s": "St_John's",
    "saint john s": "St_John's",
    "st joseph s": "St_Joseph's_PA",
    "saint joseph s": "St_Joseph's_PA",
    "st josephs": "St_Joseph's_PA",
    "saint josephs": "St_Joseph's_PA",
    "st joseph s pa": "St_Joseph's_PA",
    "saint joseph s pa": "St_Joseph's_PA",
    "st mary s": "St_Mary's_CA",
    "saint mary s": "St_Mary's_CA",
    "st francis": "St_Francis_PA",
    "saint francis": "St_Francis_PA",
    "mt st mary s": "Mt_St_Mary's",
    "mount st mary s": "Mt_St_Mary's",

    # Western Kentucky
    "western kentucky": "WKU",
    "w k u": "WKU",
    "wku": "WKU",

    # Helpful aliases
    "alabama st": "Alabama_St",
    "alabama state": "Alabama_St",
    "siu edwardsville": "SIUE",
    "texas a m": "Texas_A&M",
    "mcneese st": "McNeese_St",
    "mississippi st": "Mississippi_St",
    "uc san diego": "UC_San_Diego",
    "iowa st": "Iowa_St",
    "kansas st": "Kansas_St",
    "florida st": "Florida_St",
    "colorado st": "Colorado_St",
    "grand canyon": "Grand_Canyon",
    "north carolina": "North_Carolina",  # keep UNC distinct from NC State
}


def build_team_index(teams_df: pd.DataFrame) -> dict:
    idx: dict[str, str] = {}
    for _, r in teams_df.iterrows():
        c = canon(str(r["team_name"]).replace("_", " "))
        idx[c] = str(r["team_name"])
    return idx

def token_score(a: str, b: str) -> float:
    sa = set(a.split()); sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def parse_user_mapping_csv(text: str) -> dict:
    if not text or not text.strip():
        return {}
    mp: dict[str, str] = {}
    for line in text.strip().splitlines():
        if not line.strip() or "," not in line:
            continue
        left, right = line.split(",", 1)
        mp[left.strip()] = right.strip()
    return mp

def map_names(teams_df: pd.DataFrame, names: pd.Series, user_map_text: str) -> dict:
    idx = build_team_index(teams_df)
    alias = DEFAULT_ALIAS.copy()
    user_map = parse_user_mapping_csv(user_map_text)

    mapped: dict[str, str | None] = {}
    for name in names.unique():
        raw_name = str(name)
        c = canon(raw_name)
        # user override
        if raw_name in user_map:
            mapped[raw_name] = user_map[raw_name]
            continue
        # alias
        c_alias_raw = alias.get(c, c)
        c_alias_key = canon(c_alias_raw)
        if c_alias_key in idx:
            mapped[raw_name] = idx[c_alias_key]
            continue
        # direct
        if c in idx:
            mapped[raw_name] = idx[c]
            continue
        # fallback token overlap
        best_team = None; best_score = -1.0
        for c_key, team_str in idx.items():
            sc = token_score(c_alias_key, c_key)
            if sc > best_score:
                best_score = sc
                best_team = team_str
        mapped[raw_name] = best_team if best_score >= 0.5 else None
    return mapped

# =================== Reseeding (4 per seed, skip non-tourney) ===================
def build_model_reseed_mapping(tournament_team_set: set[str], rankings: pd.DataFrame) -> dict:
    mapping: dict[str, int] = {}
    seed = 1
    bucket_count = 0
    for _, row in rankings.sort_values(["rank"], ascending=True).iterrows():
        t = str(row["team_name"])
        if t not in tournament_team_set:
            continue
        if t in mapping:
            continue
        mapping[t] = seed
        bucket_count += 1
        if bucket_count == 4:
            seed += 1
            bucket_count = 0
        if seed > 16:
            break
    return mapping

# =================== Prob + annotation ===================
def implied_seed_from_rank(rank: int, n_teams: int) -> int:
    bin_size = n_teams / 16.0
    return int(min(max(math.ceil(rank / bin_size), 1), 16))

def logistic_prob(rA: float, rB: float, scale: float = 8.0) -> float:
    if pd.isna(rA) or pd.isna(rB):
        return float("nan")
    return 1.0 / (1.0 + math.exp(-(rA - rB) * scale))

def join_and_annotate(
    games64: pd.DataFrame,
    rankings: pd.DataFrame,
    teams_df: pd.DataFrame,
    prob_scale: float,
    close_gap: float,
    blowout_gap: float,
    upset_cut: float,
    user_map_text: str,
):
    if games64 is None or games64.empty:
        return pd.DataFrame(), []

    name_map = map_names(teams_df, pd.concat([games64["teamA"], games64["teamB"]]), user_map_text)
    out = games64.copy()
    out["teamA_model"] = out["teamA"].map(name_map)
    out["teamB_model"] = out["teamB"].map(name_map)

    model = rankings[["team_name", "rating", "rank"]]
    out = (
        out.merge(model, left_on="teamA_model", right_on="team_name", how="left")
          .rename(columns={"rating": "ratingA", "rank": "rankA"}).drop(columns=["team_name"])
          .merge(model, left_on="teamB_model", right_on="team_name", how="left")
          .rename(columns={"rating": "ratingB", "rank": "rankB"}).drop(columns=["team_name"])
    )

    # Build tournament-team set based on resolved names
    resolved_teams = set(out["teamA_model"].dropna().unique()) | set(out["teamB_model"].dropna().unique())
    reseed_map = build_model_reseed_mapping(resolved_teams, rankings)

    out["modelReseedA"] = out["teamA_model"].map(reseed_map)
    out["modelReseedB"] = out["teamB_model"].map(reseed_map)

    # (Optional) legacy implied seed display
    n_teams = len(rankings)
    out["impliedSeedA"] = out["rankA"].apply(lambda r: implied_seed_from_rank(r, n_teams) if pd.notna(r) else pd.NA)
    out["impliedSeedB"] = out["rankB"].apply(lambda r: implied_seed_from_rank(r, n_teams) if pd.notna(r) else pd.NA)

    # Probabilities
    out["pA"] = out.apply(lambda r: logistic_prob(r["ratingA"], r["ratingB"], scale=prob_scale), axis=1)
    out["pB"] = 1 - out["pA"]

    # Gaps — use model reseed vs official
    out["seedGap"] = out["seedB"] - out["seedA"]
    out["modelSeedGap"] = out["modelReseedB"] - out["modelReseedA"]
    out["gap_discrepancy"] = (out["seedGap"] - out["modelSeedGap"]).abs()

    out["closer_than_seeds"] = (out["seedGap"].abs() >= close_gap) & (
        out["modelSeedGap"].abs() < out["seedGap"].abs()
    )
    out["more_one_sided"] = (out["modelSeedGap"].abs() - out["seedGap"].abs() >= blowout_gap)

    worse_is_A = out["seedA"] > out["seedB"]
    out["upset_radar"] = False
    out.loc[worse_is_A & (out["pA"] >= upset_cut), "upset_radar"] = True
    out.loc[(~worse_is_A) & (out["pB"] >= upset_cut), "upset_radar"] = True

    unresolved = sorted([k for k, v in name_map.items() if v is None])
    return out, unresolved

# =================== UI ===================
if run_btn:
    if not (teams_file and games_file and bracket_file):
        st.error("Please upload Teams, Games, and Bracket files.")
        st.stop()

    avg_spl = parse_avg_spl(avg_spl_json_file, avg_spl_text)

    try:
        segw = ast.literal_eval(segment_weighting)
    except Exception:
        segw = [0.5, 1, 2]
    if not isinstance(segw, (list, tuple)) or not segw:
        segw = [0.5, 1, 2]

    try:
        teams_df = load_teams_df(teams_file)
    except Exception as e:
        st.error(f"Teams file error: {e}")
        st.stop()

    try:
        games_df = read_games_df(games_file)
    except Exception as e:
        st.error(f"Games file error: {e}")
        st.stop()

    try:
        bracket_df = read_bracket_df(bracket_file)
    except Exception as e:
        st.error(f"Bracket file error: {e}")
        st.stop()

    chosen_year: int | None = None
    if tournament_year_input.strip():
        try:
            chosen_year = int(tournament_year_input.strip())
        except Exception:
            st.warning("YEAR input is not an integer; auto-detecting latest.")
            chosen_year = None

    # >>> EXACT Colley <<<
    rankings_df, predictability = compute_colley_rankings(
        games_df, teams_df, avg_spl, segmentWeighting=tuple(segw), useWeighting=use_weighting
    )

    # Round of 64
    games64 = parse_round_of_64(
        bracket_df,
        year=chosen_year,
        prefer_grouping=(pair_mode.startswith("Group by"))
    )
    if games64.empty:
        st.error("Parsed 0 Round-of-64 games after YEAR filter. Verify the bracket file & pairing mode.")
        st.stop()

    # Pre-show unresolved names to help user fix mapping
    tmp_map = map_names(teams_df, pd.concat([games64["teamA"], games64["teamB"]]), user_mapping_text)
    unresolved_pre = sorted([k for k, v in tmp_map.items() if v is None])
    if unresolved_pre:
        with st.expander("⚠️ Unresolved bracket names — add to mapping overrides", expanded=True):
            st.write(pd.DataFrame({"bracket_name": unresolved_pre}))

    annotated, unresolved = join_and_annotate(
        games64,
        rankings_df,
        teams_df,
        prob_scale=prob_scale,
        close_gap=close_gap_thresh,
        blowout_gap=blowout_gap_thresh,
        upset_cut=upset_prob_cut,
        user_map_text=user_mapping_text,
    )

    st.subheader("Season summary")
    colA, colB, colC = st.columns(3)
    colA.metric("Teams", len(teams_df))
    colB.metric("Games", len(games_df))
    colC.metric("Predictability (Colley)", f"{predictability:.2f}%")

    if unresolved:
        with st.expander("⚠️ Remaining unresolved names (post-join)", expanded=False):
            st.write(pd.DataFrame({"bracket_name": unresolved}))
            st.caption("Add lines to the sidebar mapping to resolve these, then re-run.")

    tabs = st.tabs([
        "All Round of 64",
        "Upset radar",
        "Closer than seeds",
        "More one-sided",
        "Rankings (model)",
    ])

    base_cols = [
        "game_id",
        "teamA", "seedA", "teamA_model", "rankA", "modelReseedA", "ratingA", "pA",
        "teamB", "seedB", "teamB_model", "rankB", "modelReseedB", "ratingB", "pB",
        "seedGap", "modelSeedGap", "gap_discrepancy",
        "closer_than_seeds", "more_one_sided", "upset_radar",
    ]

    def _show(df: pd.DataFrame, container, *, key_suffix: str) -> None:
        if df.empty:
            container.info("No games match this filter.")
            return
        view = df[base_cols].sort_values(
            ["upset_radar", "gap_discrepancy"], ascending=[False, False]
        )
        container.dataframe(view, use_container_width=True)
        csv_buf = io.StringIO(); view.to_csv(csv_buf, index=False)
        container.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"round64_annotated_{key_suffix}.csv",
            mime="text/csv",
            key=f"dl_{key_suffix}_{uuid.uuid4().hex}",
        )

    with tabs[0]:
        _show(annotated, st, key_suffix="all")
    with tabs[1]:
        _show(annotated[annotated["upset_radar"] == True], st, key_suffix="upset")
    with tabs[2]:
        _show(annotated[annotated["closer_than_seeds"] == True], st, key_suffix="closer")
    with tabs[3]:
        _show(annotated[annotated["more_one_sided"] == True], st, key_suffix="one_sided")
    with tabs[4]:
        st.dataframe(rankings_df.sort_values("rank"), use_container_width=True)
        csv_buf2 = io.StringIO(); rankings_df.to_csv(csv_buf2, index=False)
        st.download_button(
            "Download rankings CSV",
            data=csv_buf2.getvalue(),
            file_name="model_rankings.csv",
            mime="text/csv",
            key=f"dl_rankings_{uuid.uuid4().hex}",
        )

else:
    st.info("⬅️ Upload your files, set YEAR & pairing mode, then click **Run analysis** in the sidebar.")
