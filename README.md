# Platform ranking and sequential search

Replication code for the theory simulations, descriptive evidence and prospective experiment.

## Run

Python 3.12 was used for the replication. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

This writes tables, figures and numerical results to `results/`. The full run includes 10,000 studies per power scenario, 5,000 per allocation comparison and one million simulated pairs for the payment calculation. Seeds and batch sizes match the paper. Allow several minutes, depending on the machine.

Individual parts can be run with `python reproduce.py empirical`, `model`, `rankings` or `power`. `python check_results.py` compares the outputs with reported values.

## Files

| File | Purpose |
| --- | --- |
| `ranking_simulation.py` | Seven ranking families, cumulative proximity and the local friction response; six market sizes |
| `empirical.py` | Twenty-class table, five-class table, correlations, geography summaries and the two-panel comparison |
| `experiment_model.py`, `continuous_model.py` | Finite-grid and continuous-price dynamic programmes |
| `experiment_predictions.py` | Experimental predictions, stopping thresholds and payoff accounting |
| `exact_arithmetic.py`, `verify_exact.py` | Independent rational-arithmetic calculation and comparison |
| `experiment_analysis.py` | Cell-mean contrasts, pair CR2 inference, missing-outcome bounds and permutation check |
| `experiment_power.py`, `power_helpers.py` | Behavioural planning simulations |
| `null_checks.py`, `allocation_comparison.py` | Union-null size checks and alternative friction allocations |
| `experiment_budget.py` | Participant payments, screening allowance and payout distribution |
| `randomisation.py` | Paired assignment and task-book generation |

The experiment is prospective. Simulated outcomes are not participant data. In the power code, `alpha_assumed` denotes the paper's responsiveness parameter, omega; the analysis module uses `alpha` for test size.

The default figures use Matplotlib's serif font. To render the ranking figure with the paper's New TX font, install LaTeX with `newtx` and Poppler, then run:

```sh
python ranking_simulation.py --plot-only --paper-font
```

The empirical figure reproduces the plotted quantities with a revised layout. Generated `.tex` tables require `booktabs`.

## Data

The CSVs contain the supplied aggregate exports, with numeric values preserved. `n_searches` counts results-page records, not unique people or 30-minute sessions.

| File | Rows | Aggregation and restrictions |
| --- | ---: | --- |
| `seasonal.csv` | 80 | Twenty activity classes × four sampled weeks |
| `daily.csv` | 35 | Five classes × 14–20 April 2025; positive clicks and listings, within the common geographic bounds |
| `selected.csv` | 5 | Five classes on 14 April; the same stated restrictions plus a goal-directed action and positive good use |
| `geography.csv` | 5 | Coordinate extrema and counts for the selected export |

The seasonal weeks are 14–20 April, 14–20 July and 13–19 October 2025, and 13–19 January 2026. The geographic bounds are latitude 55.49–56.02 and longitude 37.32–37.97. Means across weeks or days use record-count weights. The blank diagonal cell is a missing value in the original export; that class's diagonal uses three weeks.

Conversion is a weighted average of reported rates. Its effective denominators cannot be recovered from the exports, so it is not a verified pooled conversion probability. The saved queries support the stated filters, but their exact correspondence to every export cannot now be checked. Clicks may include repeat actions; they are not counts of distinct inspected listings. `pct_within` in the geography export is not an estimate of geographic coverage.

Individual results-page records and the coordinates behind the paper's heat maps are unavailable. The aggregate tables and geography statistics can be reproduced; the heat-map densities cannot. The data do not identify displayed rankings or inspection sequences.

## Checks

`reproduce.py model` runs the 80 model, inference, budget and randomisation tests, as well as the independent exact calculation. The ranking script checks its Bellman recursion, reach probabilities, cumulative proximity conditions, local slopes and grid accuracy. `check_results.py` checks the main reported numbers after a full run.

For a demonstration assignment book:

```sh
python randomisation.py --demo-seed 1710 --output-dir results/demo
```

Operational assignment books and secret keys should remain private. Neither is included here.
