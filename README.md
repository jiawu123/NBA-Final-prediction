# NBA Finals prediction: an evidence-first rebuild

A small research project using Basketball Reference snapshots to study NBA Finals matchups. The current dataset contains **post-Finals historical statistics**, so this repository now treats model scores as **retrospective diagnostics**, not validated forecasts.

## Read the report

**[中文图文报告：为什么应先修复数据，再优化模型](reports/NBA_PREDICTION_REPORT.zh-CN.md)**

The report includes a source-data audit, four charts, chronological model comparisons, a 2024 matchup profile, and prioritized next steps. Exact results and input checksums are in [reports/results](reports/results).

![Historical snapshots reveal the winner](reports/figures/01_outcome_leakage.png)

## What changed

- Recover the 2014 statistics hidden behind an extra empty header row.
- Preserve the unknown 2024 winner as missing and exclude it from supervised training.
- Validate one-to-one joins, finalist coverage, unique team keys, and valid winners.
- Represent each Finals as one East-versus-West feature difference; split by year before mirroring training examples.
- Fit preprocessing inside each training fold and compare simple baselines, regularized logistic regression, and a shallow random forest.
- Export inspectable results, reproducible charts, and regression tests. There is no live prediction command until trustworthy pre-Finals historical inputs are available.

These changes improve **correctness and reproducibility**. They do not establish an increase in future prediction accuracy.

## Reproduce

Tested with Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python src/NBA_PREDICTION_CODES.py
python src/plot_report.py
```

To reproduce the historical 87% result for audit only, run `python src/reproduce_legacy.py` from a clone containing the original Git history. The exact run environment is recorded in [environment.json](reports/results/environment.json).

The analysis resolves its default paths relative to the repository, so it also runs from another working directory. Optional `--data-dir` and `--output-dir` arguments select input and result directories. Plotting reads the default `reports/results` directory and writes `reports/figures`.

## Data and evaluation scope

- **31 canonical CSVs**: 15 advanced-stat tables, 15 per-100-possession tables, and one Finals label table. An identical extra 2016 file is retained but not loaded.
- **14 labeled Finals, 2010–2023**: 28 finalist-team observations, not 28 independent series. The stored 2024 matchup has two teams and an unknown winner.
- **Expanding-window diagnostics, 2015–2023**: five initial training years, then one held-out Finals per year; nine evaluation series total.
- **Three fixed features**: East-minus-West net rating, offensive rebound rate, and offensive turnover rate. These are illustrative choices, not feature-selection discoveries or causal effects.
- Historical values contain the Finals themselves, whereas the 2024 finalist records show 12 playoff wins each. A chronological split cannot repair that mismatch.

The former README's 87% accuracy and Boston 15% / Dallas 8% outputs are not retained as performance claims. The original implementation and narrative remain available in [the original commit](https://github.com/jiawu123/NBA-Final-prediction/tree/999dc8f).

## Sources

The original repository attributes the tables to [Basketball Reference](https://www.basketball-reference.com/). The [2023 playoff summary](https://www.basketball-reference.com/playoffs/NBA_2023.html) provides a spot-check of the historical table's scope. Exact download dates and pre-Finals cutoffs are not recorded in the supplied CSVs. See the report for evidence and limitations.
