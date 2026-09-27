"""Tables and descriptive comparisons from the four aggregate exports."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/empirical'
CLASSES = ('food', 'leisure', 'shopping', 'pharmacy', 'gas_stations')
METRICS = ('avg_clicks', 'conversion_rate', 'avg_answers_count',
           'avg_bounds_diag_km', 'avg_good_use_rate')


def read(name):
    with (HERE / name).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ('activity_class', 'season', 'log_date'):
                r[k] = float(v) if v else np.nan
    return rows


def weighted_mean(rows, field):
    valid = [r for r in rows if np.isfinite(r[field])]
    return float(np.average([r[field] for r in valid],
                            weights=[r['n_searches'] for r in valid]))


def correlations(rows, x='avg_clicks', y='conversion_rate'):
    a, b, w = (np.array([r[k] for r in rows]) for k in (x, y, 'n_searches'))
    da, db = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return {'pearson': float(pearsonr(a, b).statistic),
            'spearman': float(spearmanr(a, b).statistic),
            'record_weighted': float(np.sum(w*da*db) /
                                     np.sqrt(np.sum(w*da**2)*np.sum(w*db**2)))}


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(name, header, rows):
    text = ['\\begin{tabular}{l' + 'r'*(len(header)-1) + '}', '\\toprule',
            ' & '.join(header) + r' \\', '\\midrule']
    text += [' & '.join(row) + r' \\' for row in rows]
    text += ['\\bottomrule', '\\end{tabular}', '']
    (OUT / name).write_text('\n'.join(text))


def label(name):
    return name.replace('_', r'\_')


def figure(summary, daily, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    a.scatter([r['avg_clicks'] for r in summary],
              [r['conversion_rate'] for r in summary],
              s=[12 + 9*np.sqrt(r['n_searches']/1e6) for r in summary],
              facecolors='none', edgecolors='.2')
    offsets = {'transport': (-3, 12), 'pharmacy': (4, -3), 'enterprise': (-4, -12), 'shopping': (4, -2)}
    for r in summary:
        a.annotate(r['activity_class'].replace('_', ' '),
                   (r['avg_clicks'], r['conversion_rate']), xytext=offsets.get(r['activity_class'], (4, 3)),
                   textcoords='offset points', fontsize=7)
    a.set(xlabel='Mean clicks per record', ylabel='Reported conversion',
          title='(a) Four-week class averages')
    monday = {r['activity_class']: r for r in daily if r['log_date']=='2025-04-14'}
    selected = {r['activity_class']: r for r in selected}
    x = [monday[c]['avg_clicks'] for c in CLASSES]
    y = [selected[c]['avg_clicks'] for c in CLASSES]
    pos = np.arange(len(CLASSES))
    b.hlines(pos, x, y, color='.6', linewidth=1)
    b.scatter(x, pos, facecolors='white', edgecolors='.2', label='Clicked sample', zorder=3)
    b.scatter(y, pos, color='.2', label='Selected converting sample', zorder=3)
    b.set(yticks=pos, yticklabels=[c.replace('_', ' ') for c in CLASSES],
          xlabel='Mean clicks per record', title='(b) Five classes, 14 April 2025')
    b.invert_yaxis()
    b.legend(frameon=False, fontsize=8, loc='lower right')
    fig.savefig(OUT / 'class_comparisons.pdf')
    fig.savefig(OUT / 'class_comparisons.png', dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seasonal, daily, selected, geography = [read(n + '.csv') for n in
                                            ('seasonal', 'daily', 'selected', 'geography')]
    classes = sorted({r['activity_class'] for r in seasonal})
    summary = []
    for c in classes:
        rows = [r for r in seasonal if r['activity_class']==c]
        clicks = [r['avg_clicks'] for r in rows]
        summary.append({'activity_class': c, 'n_searches': int(sum(r['n_searches'] for r in rows)),
                        **{k: weighted_mean(rows, k) for k in METRICS},
                        'click_min': min(clicks), 'click_max': max(clicks),
                        'click_sd': float(np.std(clicks)),
                        'diagonal_weeks': sum(np.isfinite(r['avg_bounds_diag_km']) for r in rows)})
    summary.sort(key=lambda r: -r['avg_clicks'])
    write_csv('class_summary.csv', summary)
    table('activity_classes.tex', ['Class', 'Records (m)', 'Clicks', 'Weekly range',
                                   'Reported conv.', 'Listings', 'Diagonal (km)'],
          [[label(r['activity_class']), f"{r['n_searches']/1e6:.2f}", f"{r['avg_clicks']:.2f}",
            f"[{r['click_min']:.2f}, {r['click_max']:.2f}]", f"{r['conversion_rate']:.3f}",
            f"{r['avg_answers_count']:.1f}", f"{r['avg_bounds_diag_km']:.1f}" +
            (r'$^{*}$' if r['diagonal_weeks']<4 else '')] for r in summary])
    weeks = ('spring', 'summer', 'autumn', 'winter')
    pairs = []
    for w1, w2 in itertools.combinations(weeks, 2):
        rows = [{r['activity_class']: r for r in seasonal if r['season']==w} for w in (w1,w2)]
        pairs.append({'weeks': [w1, w2], **{m: float(spearmanr(
            [rows[0][c][m] for c in classes], [rows[1][c][m] for c in classes]).statistic)
            for m in ('avg_clicks', 'conversion_rate')}})
    five = []
    for c in CLASSES:
        rows = [r for r in daily if r['activity_class']==c]
        monday = next(r for r in rows if r['log_date']=='2025-04-14')
        sel = next(r for r in selected if r['activity_class']==c)
        five.append({'activity_class': c, 'weekly_clicks': weighted_mean(rows, 'avg_clicks'),
                     'click_min': min(r['avg_clicks'] for r in rows),
                     'click_max': max(r['avg_clicks'] for r in rows),
                     'selected_records': int(sel['n_searches']), 'selected_clicks': sel['avg_clicks'],
                     'selected_listings': sel['avg_answers'],
                     'monday_clicks': monday['avg_clicks'], 'monday_listings': monday['avg_answers'],
                     'selected_to_monday_count_ratio': sel['n_searches']/monday['n_searches'],
                     'centroid_sd_ns_km': 6371*np.pi/180*sel['std_answer_lat'],
                     'centroid_sd_ew_km': 6371*np.pi/180*np.cos(np.deg2rad(sel['avg_answer_lat']))*sel['std_answer_lon']})
    write_csv('five_class_summary.csv', five)
    table('five_classes.tex', ['Class', 'Weekly clicks', 'Daily range', 'Selected records',
                               'Selected clicks', 'Selected listings'],
          [[label(r['activity_class']), f"{r['weekly_clicks']:.2f}",
            f"[{r['click_min']:.2f}, {r['click_max']:.2f}]", f"{r['selected_records']:,}",
            f"{r['selected_clicks']:.2f}", f"{r['selected_listings']:.2f}"] for r in five])
    distances = []
    for a, b in itertools.combinations(selected, 2):
        lat1, lat2 = np.deg2rad([a['avg_answer_lat'], b['avg_answer_lat']])
        dl = np.deg2rad(b['avg_answer_lon']-a['avg_answer_lon'])
        h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2
        distances.append(float(2*6371*np.arcsin(np.sqrt(h))))
    diagnostics = {
        'seasonal_records': int(sum(r['n_searches'] for r in seasonal)),
        'daily_records': int(sum(r['n_searches'] for r in daily)),
        'selected_records': int(sum(r['n_searches'] for r in selected)),
        'class_click_sd': float(np.std([r['avg_clicks'] for r in summary])),
        'mean_within_class_click_sd': float(np.mean([r['click_sd'] for r in summary])),
        'week_pair_spearman': pairs, 'class_correlations': correlations(summary),
        'weekly_correlations': {w: correlations([r for r in seasonal if r['season']==w]) for w in weeks},
        'excluding_other': correlations([r for r in summary if r['activity_class']!='other']),
        'food_leisure_record_share': sum(r['n_searches'] for r in summary if r['activity_class'] in ('food','leisure'))/sum(r['n_searches'] for r in summary),
        'daily_clicks_exceed_listings': sum(r['avg_clicks']>r['avg_answers'] for r in daily),
        'maximum_centroid_mean_separation_km': max(distances),
        'daily_class_order': {d: [r['activity_class'] for r in sorted(
            [r for r in daily if r['log_date']==d], key=lambda r:-r['avg_clicks'])]
            for d in sorted({r['log_date'] for r in daily})},
        'conversion_noninteger_totals': sum(abs(r['conversion_rate']*r['n_searches']-round(r['conversion_rate']*r['n_searches']))>1e-6 for r in seasonal),
        'geographic_bounds': geography}
    (OUT / 'summary.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    (OUT / 'table_notes.txt').write_text(
        'Counts refer to results-page records. Means weight the exported means by record counts.\n'
        'Conversion is a weighted average of reported rates, not a verified pooled probability: effective denominators are unavailable.\n'
        '* One missing diagonal is omitted; that class uses three weeks.\n'
        'The two five-class samples use different outcome restrictions; their comparison is descriptive.\n')
    figure(summary, daily, selected)
    print(json.dumps({k: diagnostics[k] for k in ('seasonal_records','daily_records','selected_records','class_correlations')}, indent=2))


if __name__=='__main__':
    main()
