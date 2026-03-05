# Mathematics-of-Ranking

A collection of tools for analyzing team rankings and tournament matchups using **Shortest Path Length (SPL)**–based methods and probabilistic bracket modeling.

---

## Installation & Running

To run the Streamlit interface:

```bash
streamlit run <filename>
```

Replace `<filename>` with the appropriate Streamlit script.

---

## Modules

### AvgSPL

Computes **Average Shortest Path Length (SPL)** metrics between teams using game results.

#### Inputs

- **Games (.txt)**  
  List of games played.

- **Teams (.txt)**  
  List of teams included in the analysis.

#### Outputs

- **Shortest Path Between Two Teams**  
  Calculates the SPL between any pair of teams.

- **Average SPL Distribution**  
  Bar chart with bins showing the distribution of average SPL values across teams.

- **Average SPL CSV**  
  A CSV file containing the average SPL value for every team.

---

### BracketComparison

Evaluates tournament brackets and matchup probabilities using SPL rankings and configurable parameters.

#### Required Inputs

- **Teams (.txt)** – List of teams  
- **Games (.txt)** – List of game results  
- **Bracket Matchups (.csv)** – Tournament bracket structure  
- **SPL Dictionary** – Precomputed SPL values  
- **Tournament Year** – Year of the bracket being evaluated

#### Optional Inputs

- **Segment Weighting** – Adjust weighting of time ranking segments  
- **Probability Sharpness** – Controls probability curve steepness  
  - Recommended value: **~6.15**
- **Closer-than-Seeding Threshold** – Flag games closer than seeding suggests  
- **More-One-Sided Threshold** – Flag games more lopsided than seeding suggests  
- **Upset Radar Threshold** – Detect potential upset opportunities  
- **Name Mapping** – Resolve team name differences across datasets

#### Outputs

- **Raw Rankings** – Ranking of all teams based on SPL metrics  
- **Round of 64 Matchups** – Initial tournament matchups and predictions  
- **Upset Radar** – Games with high upset potential  
- **Closer-than-Seeding Games** – Matchups projected to be tighter than expected  
- **More-One-Sided Games** – Matchups projected to be more lopsided than expected  
- **Head-to-Head Probabilities** – Win probabilities for any matchup between any two teams
- **Full Bracket with Probabilities** – Complete tournament bracket with advancement probabilities

---