# Leaderboards

Experiment leaderboard artifacts are grouped by experiment family and comparison type.

## Structure

- `h4/overall/leaderboard.csv`: aggregate H4 leaderboard across model/input settings.
- `h4/per_case/`: case-wise H4 rankings and winner summaries.
- `h4/per_case/figures/`: figures derived from per-case rankings.
- `h4/input_ratio/`: model x input-ratio best-RMSE matrix and heatmap sources.
- `h4/input_ratio/figures/`: heatmap image exports.
- `archive/`: legacy backups retained for reference.

## Naming

Files inside each category use short, local names such as `top5.csv`,
`winners.csv`, and `best_rmse_matrix.csv`. The experiment family and category are
encoded by the directory path instead of repeated in every filename.
