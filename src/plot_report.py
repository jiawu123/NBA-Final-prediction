"""Build the static figures embedded in the GitHub Markdown report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from NBA_PREDICTION_CODES import ROOT

BLUE, GOLD, INK, GREY = '#2864A0', '#B97B22', '#263445', '#677586'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': GREY, 'ytick.color': INK,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    source, dest = ROOT / 'reports' / 'results', ROOT / 'reports' / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    teams = pd.read_csv(source / 'finalists.csv')
    known = teams.loc[teams.Is_Winner.notna()]
    years = sorted(known.Year.unique())
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for label, marker, color, name in [(1, 'o', BLUE, 'Champion'), (0, 's', GOLD, 'Runner-up')]:
        subset = known.loc[known.Is_Winner == label].sort_values('Year')
        ax.plot(subset.Year, subset.W, marker=marker, color=color, lw=1.5, ms=6, label=name)
    ax.axhline(12, color=GREY, ls='--', lw=1, label='Wins before Finals: 12 for each finalist')
    ax.set(xticks=years, yticks=range(10, 18), ylim=(10, 17), ylabel='Cumulative playoff wins (W)',
           title='Historical snapshots already contain the championship outcome')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='lower left', frameon=False, fontsize=10)
    fig.text(.125, -.035, '2010–2023: W = 16 identifies all 14 champions. This is outcome leakage, not predictive skill.', fontsize=10, color=GREY)
    save(fig, dest / '01_outcome_leakage.png')

    metrics = pd.read_csv(source / 'metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={'width_ratios': [1.1, 1]})
    labels = ['50/50\nreference', 'Higher net\nrating', 'Logistic\nregression', 'Random\nforest']
    colors = [GREY, GOLD, BLUE, BLUE]
    axes[0].bar(np.arange(4), metrics.accuracy, color=colors, width=.65)
    for i, row in metrics.iterrows():
        text = '50% expected' if i == 0 else f'{int(row.correct_or_expected)}/{int(row.series)}'
        axes[0].text(i, row.accuracy + .025, text, ha='center', fontsize=10)
    axes[0].set(xticks=range(4), xticklabels=labels, ylim=(0, 1.15), title='Series selection accuracy')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_yticks(np.arange(0, 1.01, .2))
    prob = metrics.loc[metrics.brier.notna()]
    axes[1].bar(range(len(prob)), prob.brier, color=[GREY, BLUE, BLUE], width=.6)
    for i, value in enumerate(prob.brier):
        axes[1].text(i, value + .008, f'{value:.3f}', ha='center', fontsize=10)
    axes[1].set(xticks=range(3), xticklabels=['50/50\nreference', 'Logistic\nregression', 'Random\nforest'],
                ylim=(0, max(.31, prob.brier.max() + .07)), title='Brier score (lower is better)')
    fig.suptitle('Walk-forward diagnostics · 9 held-out Finals, 2015–2023', fontsize=16, y=1.01)
    fig.text(.07, -.075, 'All feature snapshots still include the Finals. These scores do NOT estimate pre-Finals performance.\nThe net-rating rule selects a team only; its 0/1 decisions are not treated as probability forecasts.', fontsize=10, color=GREY)
    fig.tight_layout(w_pad=3)
    save(fig, dest / '02_retrospective_backtest.png')

    backtest = pd.read_csv(source / 'backtest.csv')
    names = ['Logistic regression', 'Random forest']
    fig, ax = plt.subplots(figsize=(11, 4.3))
    pivot = []
    test_years = sorted(backtest.Year.unique())
    for name in names:
        rows = backtest.loc[backtest.model == name].sort_values('Year')
        pivot.append(np.where(rows.East_won == 1, rows.p_east, 1 - rows.p_east))
    matrix = np.array(pivot)
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(2):
        for j in range(len(test_years)):
            ax.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center',
                    color='white' if matrix[i,j] > .65 else INK)
    ax.set(xticks=range(len(test_years)), xticklabels=test_years, yticks=range(2), yticklabels=names,
           title='Retrospective probability assigned to the actual champion')
    fig.colorbar(im, ax=ax, format=PercentFormatter(1), fraction=.03, pad=.03)
    fig.text(.125, -.03, 'Each column is one held-out year. A darker cell is not evidence of calibrated future confidence.', fontsize=10, color=GREY)
    save(fig, dest / '03_year_by_year.png')

    snapshot = teams.loc[teams.Year == 2024].set_index('Team')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, feature, title in zip(axes, ['NRtg', 'ORB%', 'OTOV%'],
                                   ['Net rating\n(points / 100 possessions)', 'Offensive rebound rate\n(%)', 'Offensive turnover rate\n(%)']):
        values = [snapshot.loc[team, feature] for team in ['Boston Celtics', 'Dallas Mavericks']]
        ax.bar(['Boston', 'Dallas'], values, color=[BLUE, GOLD], width=.55)
        for i, value in enumerate(values):
            ax.text(i, value + max(values)*.04, f'{value:g}', ha='center')
        ax.set(ylim=(0, max(values)*1.3), title=title)
    fig.suptitle('2024 stored pre-Finals snapshot · a matchup profile, not a forecast', fontsize=15, y=1.04)
    fig.text(.075, -.03, 'Boston: 12–2 in 14 games. Dallas: 12–5 in 17 games. Different opponents and sample sizes; no causal claim.', fontsize=10, color=GREY)
    fig.tight_layout(w_pad=3)
    save(fig, dest / '04_2024_matchup.png')


if __name__ == '__main__':
    main()
