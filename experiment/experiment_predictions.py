"""Generate all final model predictions, policies and directly inputtable tables."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from .experiment_model import ModelConfig,FiniteSolver,PolicyBank,payment_summary
from .continuous_model import average_continuous

HERE=Path(__file__).resolve().parent
ROOT = HERE.parent
OUT=ROOT/'results/calibration'
METRICS=['depth','depth_sd','entry_probability','conversion','exit_probability','continuation_after_first',
         'consumer_surplus','search_cost','transport_cost','revenue','welfare',
         'chosen_distance_unconditional','chosen_distance_given_purchase','price_given_purchase']


def assignment_mean_consumer_surplus(config, cells):
    """Current lottery expectation, not the equally weighted scientific contrast."""
    probabilities=np.asarray(config.data['gamma_probabilities'],dtype=float)
    if (probabilities.shape!=(len(config.data['gammas']),) or
        np.any(~np.isfinite(probabilities)) or np.any(probabilities<=0) or
        not np.isclose(probabilities.sum(),1,rtol=0,atol=1e-12)):
        raise ValueError('Invalid gamma assignment probabilities')
    by_gamma={g:p for g,p in zip(config.data['gammas'],probabilities)}
    weights=np.asarray([by_gamma[row['gamma']]/
                        (len(config.ranking_names)*len(config.cost_names)) for row in cells])
    if not np.isclose(weights.sum(),1,rtol=0,atol=1e-12):
        raise ValueError('Expected exactly one result for every treatment cell')
    return float(sum(w*row['consumer_surplus'] for w,row in zip(weights,cells)))


def json_default(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    raise TypeError(type(obj).__name__)


def write_json(path,value):path.write_text(json.dumps(value,indent=2,default=json_default)+'\n')


def write_csv(path,rows,fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def table_tex(rows,kind):
    if kind=='predictions':
        spec='llrrrr';header=r'Order & Fee & $\gamma$ & $\mathbb E[T^*]$ & Purchase & CS \\'
        body=[f"{r['ranking'].capitalize()}-first & {r['cost_points']:g} & {r['gamma']:g} & {r['depth']:.4f} & {r['conversion']:.4f} & {r['consumer_surplus']:.4f} \\\\" for r in rows]
        caption='Finite-grid model predictions by experimental condition'
        label='tab:experiment-model-predictions'
    elif kind=='accounting':
        spec='llrrrrrr';header=r'Order & Fee & $\gamma$ & SC & TC & REV & $W$ & $\mathbb E[d^*\mid\chi=1]$ \\'
        body=[f"{r['ranking'].capitalize()}-first & {r['cost_points']:g} & {r['gamma']:g} & {r['search_cost']:.4f} & {r['transport_cost']:.4f} & {r['revenue']:.4f} & {r['welfare']:.4f} & {r['chosen_distance_given_purchase']:.4f} \\\\" for r in rows]
        caption='Model-implied costs, revenue, welfare and selected distance'
        label='tab:experiment-model-accounting'
    else:
        spec='llrrrrr';header=r'Order & Fee & $\gamma$ & $\widehat m_0$ & $\widehat m_1$ & $\widehat m_2$ & $\widehat m_3$ \\'
        body=[f"{r['ranking'].capitalize()}-first & {r['cost_points']:g} & {r['gamma']:g} & "+' & '.join(f"{r[f'finite_stop_points_{t}']:.2f}" for t in range(4))+r' \\' for r in rows]
        caption='First stopping grid values at outside option 50'
        label='tab:experiment-thresholds'
    note=(r'Prices are independent uniform draws from $\{20,20.01,\ldots,120\}$; the private outside option is uniform on $\{49,49.01,\ldots,51\}$. '
          r'All monetary quantities are points; CS excludes the 40-point score endowment. '
          r'Welfare is the model accounting measure $W=\mathrm{CS}+\mathrm{REV}$.')
    if kind=='thresholds':
        note=(r'$\widehat m_t$ is the smallest grid value of best surplus at which the finite-grid policy stops after $t$ inspections. '
              r'Continue if and only if the current best surplus is strictly below this value. '
              r'For another supported outside option $u_0$, subtract $u_0-50$ from every displayed value; all remain positive. '
              r'The table is a policy prediction, not a threshold estimated from choices.')
    return '\n'.join([r'\begin{table}[htbp]',r'\singlespacing',r'\centering',r'\small',rf'\caption{{{caption}}}',rf'\label{{{label}}}',
                      rf'\begin{{tabular}}{{{spec}}}',r'\toprule',header,r'\midrule',*body,r'\bottomrule',r'\end{tabular}',
                      r'\medskip',r'\parbox{0.96\textwidth}{\footnotesize '+note+'}',r'\end{table}',''])


def main():
    start=time.perf_counter();cfg=ModelConfig.load();solver=FiniteSolver(cfg);OUT.mkdir(parents=True,exist_ok=True)
    shape=(len(cfg.ranking_names),len(cfg.cost_names),len(cfg.data['gammas']),len(cfg.outside_units),cfg.data['n_sellers'])
    all_stops=np.empty(shape,dtype=np.int64)
    finite=[];continuous=[];thresholds=[]
    for ri,r in enumerate(cfg.ranking_names):
        for ci,c in enumerate(cfg.cost_names):
            for gi,g in enumerate(cfg.data['gammas']):
                ref=average_continuous(cfg,r,c,g)
                descriptor=dict(ranking=r,cost_label=c,cost_points=cfg.data['inspection_costs_points'][c],gamma=g)
                continuous.append(dict(**descriptor,distribution='continuous_reference',**ref))
                expected=[]
                for ui,u in enumerate(cfg.outside_units):
                    policy=solver.solve_policy(r,c,g,int(u));all_stops[ri,ci,gi,ui]=policy.stop_state_units
                    expected.append(solver.expected_outcomes(policy))
                    row=dict(**descriptor,outside_units=int(u),outside_points=float(u*cfg.unit))
                    for t in range(cfg.data['n_sellers']):
                        row[f'finite_stop_units_{t}']=int(policy.stop_state_units[t])
                        row[f'finite_stop_points_{t}']=float(policy.stop_state_units[t]*cfg.unit)
                        row[f'continuous_threshold_points_{t}']=float(cfg.data['value_points']-u*cfg.unit-ref['generalised_price_cutoffs'][t])
                    thresholds.append(row)
                avg={k:float(np.mean([x[k] for x in expected])) for k in METRICS if k not in ['depth_sd','chosen_distance_given_purchase','price_given_purchase']}
                avg['depth_sd']=float(np.sqrt(np.mean([x['depth_sd']**2+x['depth']**2 for x in expected])-avg['depth']**2))
                avg['chosen_distance_given_purchase']=avg['chosen_distance_unconditional']/avg['conversion']
                avg['price_given_purchase']=avg['revenue']/avg['conversion']
                avg['reach']=np.mean([x['reach'] for x in expected],axis=0).tolist()
                avg['max_search_threshold_tie_mass']=max(max(x['search_threshold_tie_mass']) for x in expected)
                avg['max_positive_seller_tie_encounter_mass']=max(max(x['positive_seller_tie_encounter_mass']) for x in expected)
                avg['min_absolute_gain_points']=min(x['minimum_absolute_gain_points'] for x in expected)
                avg['depth_range_over_types']=[min(x['depth'] for x in expected),max(x['depth'] for x in expected)]
                rho=solver.reservation_price(descriptor['cost_points'])
                avg['finite_reservation_price']=rho
                avg['worst_type_activity_margin_finite']=cfg.data['value_points']-cfg.data['outside_max_points']-g*max(cfg.data['distances'])-rho
                avg['continuous_activity_margin']=cfg.data['value_points']-cfg.data['outside_max_points']-g*max(cfg.data['distances'])-ref['rho'][-1]
                avg['continuous_strictness_margin']=min(ref['rho'][1]-cfg.data['price_min_points'],cfg.data['price_max_points']-ref['rho'][1])-g*(max(cfg.data['distances'])-min(cfg.data['distances']))
                finite.append(dict(**descriptor,distribution='implemented_finite_grid',**avg))
    bank=PolicyBank(cfg,all_stops);bank.save(OUT/'policy_thresholds.npz')
    fields=['ranking','cost_label','cost_points','gamma','distribution']+METRICS
    write_csv(OUT/'finite_cells.csv',finite,fields)
    write_csv(OUT/'continuous_reference_cells.csv',continuous,[f for f in fields if f!='depth_sd'])
    write_json(OUT/'finite_cells.json',finite);write_json(OUT/'continuous_reference_cells.json',continuous)
    write_csv(OUT/'thresholds_by_type.csv',thresholds,list(thresholds[0]));write_json(OUT/'thresholds_by_type.json',thresholds)
    midpoint=(cfg.data['outside_min_points']+cfg.data['outside_max_points'])/2
    middle=[r for r in thresholds if r['outside_points']==midpoint]
    (OUT/'prediction_cells.tex').write_text(table_tex(finite,'predictions'))
    (OUT/'accounting_cells.tex').write_text(table_tex(finite,'accounting'))
    (OUT/'stopping_thresholds.tex').write_text(table_tex(middle,'thresholds'))
    summary=payment_summary(cfg,assignment_mean_consumer_surplus(cfg,finite))
    p=cfg.data['payment'];endow=p['score_endowment_points'];v=cfg.data['value_points']
    cmin=min(cfg.data['inspection_costs_points'].values());cmax=max(cfg.data['inspection_costs_points'].values())
    maxtransport=max(cfg.data['gammas'])*max(cfg.data['distances']);n=cfg.data['n_sellers']
    summary.update(all_action_score_min=endow+v-cfg.data['price_max_points']-maxtransport-cmax*n,
                   all_action_score_max=endow+v-cfg.data['price_min_points']-cmin,
                   exit_score_min=endow+cfg.data['outside_min_points']-cmax*n,
                   exit_score_max=endow+cfg.data['outside_max_points'],
                   actual_payment_euros=[p['fixed_euros'],p['fixed_euros']+p['prize_euros']],
                   formula='3 + 12 * E[z] / 200; E[z]=40+E[CS], equal ranking/fee weights and registered gamma probabilities.',
                   gamma_probabilities=cfg.data['gamma_probabilities'],
                   unstarted_slot_score=p['unstarted_slot_score'],
                   interpretation='Expected payment for full completion under gamma probabilities 0.4/0.2/0.4; individual realised cell counts need not be balanced. No lottery is drawn by this package.')
    write_json(OUT/'payment_summary.json',summary)
    metadata=dict(numerical_method='Exact finite-state expectations; float64 probability arithmetic; integer monetary states.',
                  n_finite_cells=len(finite),n_outside_types=len(cfg.outside_units),n_price_values=len(cfg.prices_units),
                  no_behavioural_data=True,numpy_version=np.__version__,seconds=time.perf_counter()-start,
                  configured_terminal_ties=cfg.data['terminal_tie_rule'],
                  maximum_depth_range_over_types=max(r['depth_range_over_types'][1]-r['depth_range_over_types'][0] for r in finite),
                  maximum_search_threshold_tie_mass=max(r['max_search_threshold_tie_mass'] for r in finite),
                  minimum_continuous_activity_margin=min(r['continuous_activity_margin'] for r in finite),
                  minimum_continuous_strictness_margin=min(r['continuous_strictness_margin'] for r in finite))
    write_json(OUT/'verification_summary.json',metadata)
    print(json.dumps(dict(payment=summary,verification=metadata),indent=2))


if __name__=='__main__':main()
