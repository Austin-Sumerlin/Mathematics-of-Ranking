import streamlit as st
import pandas as pd
import numpy as np
import datetime


# ============================================================
# 1. Utility from your notebook
# ============================================================

def convertDateToMassey(dateToConvert): 
    # Note, there are 718528 days between January 1, 1970 and (what would be) January 1, year 0. 
    return (dateToConvert - datetime.datetime(1970,1,1)).days + 719528


# ============================================================
# 2. Core notebook logic, parameterized by three files
# ============================================================

def run_massey_notebook_style(teams_df: pd.DataFrame,
                              games_all_df: pd.DataFrame,
                              games_notourney_df: pd.DataFrame):
    """
    This is a faithful port of your notebook logic.

    teams_df          : content of Teams file (ID, Name), IDs 1-based
    games_all_df      : content of AllGames file (includes tournament), IDs 1-based
    games_notourney_df: content of Games file (no tournament games), IDs 1-based
    """

    # -----------------------------
    # teamNames (exactly as notebook)
    # -----------------------------
    teamNames = teams_df.copy()
    teamNames.loc[:, 0] = teamNames.loc[:, 0] - 1
    numTeams = len(teamNames)

    # -----------------------------
    # masseyRanking from notebook
    # -----------------------------

    def masseyRanking():
        ##### columns of games are:
        #   column 0 = days since 1/1/0000
        #   column 1 = date in YYYYMMDD format
        #
        #   column 2 = team1 index (1-based in file)
        #   column 3 = team1 homefield (1 = home, -1 = away, 0 = neutral)
        #   column 4 = team1 score
        #   column 5 = team2 index
        #   column 6 = team2 homefield (1 = home, -1 = away, 0 = neutral)
        #   column 7 = team2 score

        games = games_notourney_df.copy()
        numGames = len(games)

        import numpy as np
        from math import ceil  # imported in your notebook (not strictly needed for core loop)

        # Massey matrix (graph Laplacian) and RHS vector of point differentials
        M = np.zeros((numTeams, numTeams), dtype=float)
        p = np.zeros(numTeams, dtype=float)

        dayBeforeSeason = games.loc[0, 0] - 1
        lastDayOfSeason = games.loc[len(games) - 1, 0]

        for i in range(numGames):
            # NOTE: your notebook converts to 0-based *here*:
            team1ID = int(games.loc[i, 2]) - 1
            team1Loc = games.loc[i, 3]
            team1Score = float(games.loc[i, 4])

            team2ID = int(games.loc[i, 5]) - 1
            team2Loc = games.loc[i, 6]
            team2Score = float(games.loc[i, 7])

            currentDay = games.loc[i, 0]

            # Time/home weighting (you set this to 1 in the notebook)
            timeWeight = 1
            weight = timeWeight

            # Update Massey matrix (degrees and off-diagonals)
            M[team1ID, team1ID] += weight
            M[team2ID, team2ID] += weight
            M[team1ID, team2ID] -= weight
            M[team2ID, team1ID] -= weight

            # Update RHS with point differential
            diff = (team1Score - team2Score) * weight
            p[team1ID] += diff
            p[team2ID] -= diff

        # The Massey matrix is singular; impose sum of ratings = 0 constraint
        M_mod = M.copy()
        p_mod = p.copy()
        M_mod[-1, :] = 1.0
        p_mod[-1] = 0.0

        ratings, *_ = np.linalg.lstsq(M_mod, p_mod, rcond=None)
        return ratings

    # -----------------------------
    # Load ALL games (exactly like notebook)
    # -----------------------------
    # columns of games are:
    #   column 0 = days since 1/1/0000
    #   column 1 = date in YYYYMMDD format
    #   column 2 = team1 index
    #   column 3 = team1 homefield (1 = home, -1 = away, 0 = neutral)
    #   column 4 = team1 score
    #   column 5 = team2 index
    #   column 6 = team2 homefield (1 = home, -1 = away, 0 = neutral)
    #   column 7 = team2 score
    games = games_all_df.copy()
    games.loc[:, 2] = games.loc[:, 2] - 1
    games.loc[:, 5] = games.loc[:, 5] - 1
    numGames = len(games)

    # -----------------------------
    # Selection Sunday list (from your notebook)
    # -----------------------------
    selectionSundayList = [
        '03/10/2002','03/16/2003','03/14/2004','03/13/2005','03/12/2006',
        '03/11/2007','03/16/2008','03/15/2009','03/14/2010','03/13/2011',
        '03/11/2012','03/17/2013','03/16/2014','03/15/2015','03/13/2016',
        '03/12/2017','03/11/2018','3/17/2019','3/15/2020','3/14/2021'
    ]

    yearOfTournament = int(str(games.iloc[-1, 1])[0:4])
    selectionSundayDate = datetime.datetime.strptime(
        selectionSundayList[yearOfTournament - 2002],
        "%m/%d/%Y"
    )
    selectionSundayInt = convertDateToMassey(selectionSundayDate)

    numberGamesBeforeMadness = len(np.where(games[0] <= selectionSundayInt)[0])

    # =====================================================
    # Tournament teams detection (your exact code)
    # =====================================================
    marchMadnessTeams = {games.iloc[-1, 2], games.iloc[-1, 5]}

    if games.iloc[-1, 4] > games.iloc[-1, 7]:
        nationalChampion = games.iloc[-1, 2]
    else:
        nationalChampion = games.iloc[-1, 5]

    finalTwo = set(marchMadnessTeams)

    for i in reversed(range(numGames)): 
        teamsInGame = {games.iloc[i, 2], games.iloc[i, 5]}
        if len(teamsInGame & marchMadnessTeams) > 0:        
            marchMadnessTeams = marchMadnessTeams.union(teamsInGame)

        if len(marchMadnessTeams) == 4:
            finalFour = set(marchMadnessTeams)
        elif len(marchMadnessTeams) == 8:
            eliteEight = set(marchMadnessTeams)
        elif len(marchMadnessTeams) == 16:
            sweet16 = set(marchMadnessTeams)
        elif len(marchMadnessTeams) == 32:
            round32 = set(marchMadnessTeams)

        if len(marchMadnessTeams) >= 64:
            indexToStart = i
            break

    marchMadnessTeamsByRoundList = []
    marchMadnessTeamsByRoundList.append(list(marchMadnessTeams))  # 64
    marchMadnessTeamsByRoundList.append(list(round32))
    marchMadnessTeamsByRoundList.append(list(sweet16))
    marchMadnessTeamsByRoundList.append(list(eliteEight))
    marchMadnessTeamsByRoundList.append(list(finalFour))
    marchMadnessTeamsByRoundList.append(list(finalTwo))
    marchMadnessTeamsByRoundList.append([nationalChampion])

    # =====================================================
    # Compute ratings exactly like notebook
    # =====================================================
    r = masseyRanking()

    # =====================================================
    # Bracket simulation + ESPN score (exact code)
    # =====================================================
    marchMadnessTeamsList = list(marchMadnessTeams)
    marchMadnessCorrect = np.zeros(len(marchMadnessTeams))

    # Create initial list of pairings as teams by themselves.  
    initList = []
    for i in range(len(marchMadnessTeamsList)):
         initList.append([marchMadnessTeamsList[i]])
    pairingsList = [initList]

    # We start with no correct predictions! 
    madnessCorrect = np.zeros(len(marchMadnessTeams))

    # Skip play-in games 
    thursdayOfMadness = selectionSundayInt + 4
    numberGamesBeforeFirstRound = len(np.where(games[0] < thursdayOfMadness)[0])

    currentRoundList = []
    currentRound = 0
    teamsInRound = [64, 32, 16, 8, 4, 2]
    gamesInRound = 0

    # First round contains every team 
    roundByRoundTeamsList = []
    currentRoundByRoundTeamsList = []

    currentPairingsList = pairingsList[-1]

    for i in range(numberGamesBeforeFirstRound, numGames): 
        team1ID = games.loc[i, 2] 
        team1Score = games.loc[i, 4]
        team2ID = games.loc[i, 5] 
        team2Score = games.loc[i, 7]    

        # Are March Madness teams in the game? Then, it's a Madness game
        if len({team1ID, team2ID} & marchMadnessTeams) == 2:
            # Figure out who would play in your bracket! 
            gamesInRound += 1 

            for k in range(len(currentPairingsList)):
                if team1ID in currentPairingsList[k]:
                    team1Teams = np.array(currentPairingsList[k])
                elif team2ID in currentPairingsList[k]:
                    team2Teams = np.array(currentPairingsList[k])

            # Find teams predicted to play, which have maximum rating 
            # in the list of teams who have played to this point
            predictedTeam1ID = team1Teams[np.argmax(r[team1Teams])]
            predictedTeam2ID = team2Teams[np.argmax(r[team2Teams])]
            currentRoundByRoundTeamsList.append(predictedTeam1ID)
            currentRoundByRoundTeamsList.append(predictedTeam2ID)

            currentRoundList.append(list(team1Teams) + list(team2Teams))

            if r[predictedTeam1ID] > r[predictedTeam2ID]:
                predictedWinner = predictedTeam1ID
            else: 
                predictedWinner = predictedTeam2ID

            if team1Score > team2Score:
                actualWinner = team1ID
            else: 
                actualWinner = team2ID

            if actualWinner == predictedWinner: 
                indexOfPick = marchMadnessTeamsList.index(actualWinner)
                madnessCorrect[indexOfPick] += 1

            if (gamesInRound == teamsInRound[currentRound] / 2):
                currentRound += 1
                pairingsList.append(currentRoundList)
                currentRoundList = []
                gamesInRound = 0
                currentPairingsList = pairingsList[-1]
                roundByRoundTeamsList.append(currentRoundByRoundTeamsList)
                currentRoundByRoundTeamsList = []

    espnScore = 0
    espnPoints = [0, 10, 30, 70, 150, 310, 630]
    for i in range(len(espnPoints)): 
        espnScore += espnPoints[i] * len(np.where(madnessCorrect == i)[0])                

    # convert team names to a convenient series
    team_name_series = teamNames.iloc[:, 1].copy()
    team_name_series.index = teamNames.iloc[:, 0]

    return (
        r,
        team_name_series,
        marchMadnessTeams,
        marchMadnessTeamsByRoundList,
        madnessCorrect,
        espnScore,
        roundByRoundTeamsList,
        espnPoints,
    )


# ============================================================
# 3. Streamlit UI
# ============================================================

def main():
    st.title("Original Massey NCAA Backtest (Notebook-Exact)")

    st.write(
        """
        This app runs **the exact same logic as your original notebook**:

        - Ratings from the *Games-without-tournament* file
        - Tournament field and rounds inferred from the *AllGames* file
        - Bracket simulated round-by-round using those ratings
        - ESPN score computed with `[0,10,30,70,150,310,630]`
        """
    )

    teams_file = st.file_uploader("Teams file (e.g. 2015Teams.txt)", type=["txt", "csv"])
    games_notourney_file = st.file_uploader("Games WITHOUT tournament (e.g. 2015Games.txt)", type=["txt", "csv"])
    games_all_file = st.file_uploader("Games WITH tournament (e.g. 2015AllGames.txt)", type=["txt", "csv"])

    if teams_file and games_notourney_file and games_all_file:
        if st.button("Run original Massey backtest"):
            try:
                teams_df = pd.read_csv(teams_file, header=None)
                games_notourney_df = pd.read_csv(games_notourney_file, header=None)
                games_all_df = pd.read_csv(games_all_file, header=None)

                (r,
                 team_names,
                 marchTeams,
                 marchTeamsByRound,
                 madnessCorrect,
                 espnScore,
                 roundByRoundTeamsList,
                 espnPoints) = run_massey_notebook_style(
                    teams_df,
                    games_all_df,
                    games_notourney_df,
                )

                st.success(f"ESPN Score (same formula as notebook): **{espnScore}**")

                # Show tournament team names (the core thing you’re debugging)
                st.subheader("Detected Tournament Teams (64-team field)")
                marchTeams_list = sorted(list(marchTeams))
                marchTeams_names = [team_names.get(t, f"ID {t}") for t in marchTeams_list]
                st.write(", ".join(marchTeams_names))

                # Show distribution of correct picks per team
                st.subheader("Correct predictions per team")
                rows = []
                for i in range(len(espnPoints)):
                    rows.append({
                        "Correct picks": i,
                        "Number of teams": int(np.sum(madnessCorrect == i)),
                        "ESPN pts per team": espnPoints[i]
                    })
                dist_df = pd.DataFrame(rows)
                st.table(dist_df)

                # Show ranking of tournament teams only (like notebook)
                st.subheader("Massey ranking of tournament teams")
                iSort = np.argsort(-r)
                rank_rows = []
                rank = 1
                for idx in iSort:
                    if idx in marchTeams:
                        rank_rows.append({
                            "Rank": rank,
                            "Team": team_names.get(idx, f"ID {idx}"),
                            "Rating": float(r[idx]),
                            "Team ID": idx,
                        })
                        rank += 1
                rank_df = pd.DataFrame(rank_rows).set_index("Rank")
                st.dataframe(rank_df, use_container_width=True)

            except Exception as e:
                st.error(f"Something went wrong while running the backtest: {e}")

    else:
        st.info("Upload all three files above, then click **Run original Massey backtest**.")


if __name__ == "__main__":
    main()
