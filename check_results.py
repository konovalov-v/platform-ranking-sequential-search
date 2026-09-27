"""Compare regenerated outputs with values printed in the paper."""
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    expected = read(HERE / 'paper_values.json')
    cells = read(OUT / 'calibration/finite_cells.json')
    thresholds = csv_rows(OUT / 'calibration/thresholds_by_type.csv')
    for table, fields in [('predictions', ['depth','conversion','consumer_surplus']),
                           ('accounting', ['search_cost','transport_cost','revenue','welfare','chosen_distance_given_purchase'])]:
        for row in expected[table]:
            rank = 'nearest' if row[0]=='Nearest-first' else 'farthest'
            fee, gamma = int(row[1]), int(row[2])
            r = next(r for r in cells if (r['ranking'],r['cost_points'],r['gamma'])==(rank,fee,gamma))
            values = [f'{r[k]:.4f}' for k in fields]
            if table=='predictions':
                t = next(t for t in thresholds if (t['ranking'],int(t['cost_points']),int(t['gamma']),int(t['outside_units']))==(rank,fee,gamma,5000))
                values += [f"{float(t[f'finite_stop_points_{j}']):.2f}" for j in range(4)]
            assert values==row[3:], (table,row,values)
    summary = csv_rows(OUT / 'empirical/class_summary.csv')
    for row in expected['classes']:
        r = next(r for r in summary if r['activity_class']==row[0])
        values = [f"{float(r['n_searches'])/1e6:.2f}", f"{float(r['avg_clicks']):.2f}",
                  f"[{float(r['click_min']):.2f}, {float(r['click_max']):.2f}]",
                  f"{float(r['conversion_rate']):.3f}", f"{float(r['avg_answers_count']):.1f}",
                  f"{float(r['avg_bounds_diag_km']):.1f}"]
        assert values==row[1:], (row,values)
    e = read(OUT / 'empirical/summary.json')
    assert e['daily_records']==3106641 and e['selected_records']==11070
    assert e['daily_clicks_exceed_listings']==16
    assert round(e['class_correlations']['pearson'],3)==-.087
    assert round(e['class_correlations']['spearman'],3)==-.063
    assert round(e['class_correlations']['record_weighted'],3)==.220
    power = read(OUT / 'power/final_power_results.json')
    for group, rates in [('base', [1,.989,.7364,.1307,.0031]),
                         ('strong_persistence', [.6557,.1152,.0023])]:
        actual = [r['primary_joint_signs']['rate'] for r in power[group]['scenarios'].values()]
        assert actual==rates, (group,actual)
    allocations = read(OUT / 'allocation/comparison_results.json')
    for key, expected_rate in [('uniform',.6336),('endpoint_40',.7378),('endpoint_425',.7644)]:
        actual = allocations['allocations'][key]['scenarios']['alpha_0.5']['joint_sign_IUT_unadjusted']['rate']
        assert actual==expected_rate, (key,actual)
    null = read(OUT / 'null/results.json')
    assert [v['joint_primary_rejection']['rate'] for v in null['cases'].values()]==[.0488,.0543,.0522,0]
    b = read(OUT / 'budget/budget_results.json')
    assert round(b['exact_expected_total_euros'],2)==2853.03
    assert round(b['screening']['expected_total_including_screening_euros'],2)==2928.03
    dist = b['direct_conditional_average_distribution']
    assert round(100*dist['probability_total_strictly_above_3000_euros'],2)==6.97
    assert list(dist['quantiles_euros'].values())==[2988,3024,3096]
    rankings = read(OUT / 'rankings/simulation_lower_costs_results.json')
    curves = {(r['N'], r['ranking_key'], round(r['gamma'],12)): r
              for m in rankings['results'] for r in m['rows']}
    for ref in expected['ranking_curves']:
        r = curves[(ref['N'],ref['ranking_key'],round(ref['gamma'],12))]
        for k in ('mean_depth','integrated_mean','local_derivative_at_zero'):
            assert math.isclose(r[k],ref[k],rel_tol=1e-10,abs_tol=1e-10), (ref,k,r[k])
    print('All reported-value checks passed.')


if __name__=='__main__':
    main()
