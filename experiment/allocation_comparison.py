#!/usr/bin/env python3
"""Pre-data design comparison: only the iid gamma assignment law is changed."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / 'power_helpers.py'
spec = importlib.util.spec_from_file_location('allocation_frozen_power', SOURCE)
power = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = power
spec.loader.exec_module(power)
ALLOCATIONS = {'uniform': (1/3, 1/3, 1/3),
               'endpoint_40': (.4, .2, .4),
               'endpoint_425': (.425, .15, .425)}


def draw_design(rng, batch, probabilities):
    """Exact original draw order. Reviewed delta: gamma probabilities alone."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (3,) or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError('Require three strictly positive gamma probabilities summing to one')
    c = rng.integers(0, 2, (batch, power.P, power.J))
    if np.array_equal(probabilities, np.full(3, 1/3)):
        g = rng.integers(0, 3, (batch, power.P, power.J))
    else:
        g = rng.choice(3, size=(batch, power.P, power.J), p=probabilities)
    u = rng.integers(4900, 5101, (batch, power.P, 1))
    prices = rng.integers(2000, 12001, (batch, power.P, power.J, 4))
    return c, g, u, prices


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(seed):
    original = power.draw_design(np.random.default_rng(seed), 3)
    uniform = draw_design(np.random.default_rng(seed), 3, ALLOCATIONS['uniform'])
    assert all(np.array_equal(a, b) for a, b in zip(original, uniform))
    original_y = power.behavioural_outcomes(np.random.default_rng(seed+1), original, .5)
    uniform_y = power.behavioural_outcomes(np.random.default_rng(seed+1), uniform, .5)
    assert np.array_equal(original_y, uniform_y)
    observation = np.random.default_rng(seed+2).random((3, power.P, 2)) >= .1
    first = power.analyse_batch(original_y, power.geometry(original[0], original[1], observation))
    second = power.analyse_batch(uniform_y, power.geometry(uniform[0], uniform[1], observation))
    assert all(np.array_equal(first[k], second[k]) for k in first)
    reference_errors = []
    for allocation in ('endpoint_40', 'endpoint_425'):
        rng = np.random.default_rng(seed+3)
        design = draw_design(rng, 2, ALLOCATIONS[allocation])
        outcomes = power.behavioural_outcomes(rng, design, .5)
        observed = rng.random((2, power.P, 2)) >= .1
        analysed = power.analyse_batch(outcomes, power.geometry(design[0], design[1], observed))
        for trial in range(2):
            data = power.reference_data(design, outcomes, observed, trial)
            for q, kind in enumerate(('interaction', 'near_simple', 'far_simple')):
                reference = power.reference.cr2(data, 'T', power.reference.contrast(power.J, kind), power.reference.Config())
                for name in ('estimate', 'se', 'df'):
                    error = abs(reference[name] - analysed[name][trial, q])
                    reference_errors.append(error)
                    assert np.isclose(reference[name], analysed[name][trial, q], rtol=1e-10, atol=1e-10)
    return {'uniform_original_design_outcomes_and_analysis_identical': True,
            'nonuniform_full_reference_numeric_checks': len(reference_errors),
            'maximum_absolute_reference_error': max(reference_errors)}


def rate(values):
    values = np.asarray(values, dtype=bool)
    p = float(values.mean())
    return {'rate': p, 'successes': int(values.sum()), 'replications': len(values),
            'Monte_Carlo_SE': math.sqrt(p*(1-p)/len(values))}


def run_allocation(name, alphas, repetitions, batch_size, seed, output_name=None):
    output_name = output_name or name
    probabilities = ALLOCATIONS[name]
    rng = np.random.default_rng(seed)
    pieces = {alpha: [] for alpha in alphas}
    occupancy = []
    counts = np.zeros(3, dtype=np.int64)
    realised_loss = []
    started = time.time()
    for offset in range(0, repetitions, batch_size):
        batch = min(batch_size, repetitions-offset)
        design = draw_design(rng, batch, probabilities)
        c, g, u, prices = design
        cached = [(power.policy_depth(prices, u, c, g, a),
                   power.policy_depth(prices, u, c, np.zeros_like(g), a)) for a in range(2)]
        observed = rng.random((batch, power.P, 2)) >= .1
        geom = power.geometry(c, g, observed)
        full_counts = geom['n'].reshape(batch, 2, 2, 3, power.J)
        middle = full_counts[:, :, :, 1]
        all_counts = full_counts.reshape(batch, -1)
        occupancy.append({'middle_any_empty': np.any(middle == 0, axis=(1,2,3)),
                          'middle_any_singleton': np.any(middle == 1, axis=(1,2,3)),
                          'middle_any_below_two': np.any(middle < 2, axis=(1,2,3)),
                          'middle_minimum_count': middle.min(axis=(1,2,3)),
                          'all_any_below_two': np.any(all_counts < 2, axis=1)})
        counts += np.bincount(g.ravel(), minlength=3)
        realised_loss.append(1-observed.mean(axis=(1,2)))
        for alpha in alphas:
            outcomes = power.behavioural_outcomes(rng, design, alpha, cached)
            result = power.analyse_batch(outcomes, geom)
            # Complete-choice payment expectation, not a dropout payment model.
            result['complete_dgp_expected_cash_per_person'] = 3+.06*(40+outcomes[2].mean(axis=(1,2,3)))
            pieces[alpha].append(result)
        if offset % (batch_size*10) == 0:
            print(json.dumps({'allocation': name, 'completed': offset+batch,
                              'replications': repetitions, 'alphas': alphas,
                              'elapsed_seconds': round(time.time()-started,2)}), flush=True)
    occupied = {k: np.concatenate([x[k] for x in occupancy]) for k in occupancy[0]}
    summary = {'gamma_probabilities': probabilities, 'seed': seed,
               'replications_per_alpha': repetitions,
               'gamma_assignment_counts': counts.tolist(),
               'mean_whole_person_loss': float(np.concatenate(realised_loss).mean()),
               'occupancy': {k: rate(v) for k,v in occupied.items() if 'count' not in k},
               'middle_minimum_count_quantiles': np.quantile(occupied['middle_minimum_count'], [0,.01,.05,.5,1]).tolist(),
               'scenarios': {}, 'runtime_seconds': time.time()-started}
    arrays = {}
    for alpha in alphas:
        result = {k: np.concatenate([p[k] for p in pieces[alpha]]) for k in pieces[alpha][0]}
        estimates = result['estimate']
        cash = result['complete_dgp_expected_cash_per_person']
        row = {'attenuation_alpha_assumed': alpha,
               'joint_sign_IUT_unadjusted': rate(result['raw_family_p'][:,0] < .05),
               'positive_interaction': rate(result['primary_p'] < .05),
               'interaction_nonestimable': rate(~result['valid'][:,0]),
               'joint_sign_component_nonestimable': rate(~np.all(result['valid'][:,1:3], axis=1)),
               'middle_gamma_conversion_contrast_nonestimable': rate(~result['valid'][:,5]),
               'effect_names': list(power.EFFECT_NAMES),
               'estimated_effect_means': estimates.mean(axis=0).tolist(),
               'estimator_sd': estimates.std(axis=0, ddof=1).tolist(),
               'mean_estimated_se': result['se'].mean(axis=0).tolist(),
               'mean_degrees_of_freedom': result['df'].mean(axis=0).tolist(),
               'complete_dgp_expected_cash_per_person': float(cash.mean()),
               'cash_mean_Monte_Carlo_SE': float(cash.std(ddof=1)/math.sqrt(repetitions))}
        summary['scenarios'][f'alpha_{alpha:g}'] = row
        for key, value in result.items():
            arrays[f'alpha_{alpha:g}__{key}'] = value
    (OUT/f'{output_name}_results.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(OUT/f'{output_name}_SIMULATION_SUMMARIES.npz', **arrays)
    print(json.dumps({'allocation_complete': name, 'runtime_seconds': summary['runtime_seconds']}), flush=True)
    return summary


def payment_and_counts():
    cells = json.loads((ROOT/'results/calibration/finite_cells.json').read_text())
    sample = cells[0]
    # Explicit source schema validation keeps payoff and lottery money separate.
    assert 'consumer_surplus' in sample, list(sample)
    by_gamma = {g: float(np.mean([x['consumer_surplus'] for x in cells if x['gamma'] == g])) for g in (0,2,4)}
    results = {}
    for name, probs in ALLOCATIONS.items():
        mean_cs = sum(p*by_gamma[g] for p,g in zip(probs, (0,2,4)))
        payment = 3+.06*(40+mean_cs)
        results[name] = {'optimal_CS_by_gamma': by_gamma,
                         'optimal_expected_payment_per_person': payment,
                         'optimal_expected_total_participant_payment': 300*payment,
                         'expected_person_rounds_per_fee_gamma_cell': [power.J*.5*p for p in probs],
                         'expected_observed_full_rank_fee_gamma_period_count': [power.P*.9*.5*p for p in probs],
                         'expected_assigned_full_rank_fee_gamma_period_count': [power.P*.5*p for p in probs]}
    base = results['uniform']['optimal_expected_total_participant_payment']
    for value in results.values():
        value['expected_total_payment_change_from_uniform'] = value['optimal_expected_total_participant_payment']-base
    return results


OUT = ROOT / "results/allocation"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--replications', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=1710)
    ap.add_argument('--validation-only', action='store_true')
    args = ap.parse_args()
    paths = [SOURCE, HERE/'experiment_analysis.py',
             HERE/'experiment_model.py',
             ROOT/'results/calibration/policy_thresholds.npz',
             ROOT/'results/calibration/finite_cells.json']
    before = {str(p.relative_to(ROOT)): hash_file(p) for p in paths}
    validation = validate(args.seed+700)
    arithmetic = payment_and_counts()
    print(json.dumps({'validation': validation, 'payment_and_counts': arithmetic}), flush=True)
    if args.validation_only:
        return
    all_results = {}
    for allocation in ALLOCATIONS:
        all_results[allocation] = run_allocation(allocation, (.5,.75), args.replications, args.batch, args.seed)
    candidates = ('endpoint_40', 'endpoint_425')
    best = max(candidates, key=lambda n: all_results[n]['scenarios']['alpha_0.5']['joint_sign_IUT_unadjusted']['rate'])
    # Separate fresh seed; a planning comparison, not participant-data selection.
    null = run_allocation(best, (0.,), args.replications, args.batch, args.seed+1,
                          output_name=best+'_null')
    after = {str(p.relative_to(ROOT)): hash_file(p) for p in paths}
    assert before == after
    out = {'status': 'PRE-DATA design comparison; alpha is assumed, not estimated responsiveness',
           'reviewed_change': 'Only iid gamma draw probabilities change; finite policy bank, behavioural mixture, fee draw, N, J, pairing, person-MCAR and CR2 helper functions are reused unchanged.',
           'validation': validation, 'source_hashes_before_and_after_identical': before,
           'design': {'participants': 300, 'pairs':150, 'rounds':24, 'fees':[1,4], 'fee_probabilities':[.5,.5],
                      'gammas':[0,2,4], 'whole_person_MCAR':.1, 'attention_kappa':5, 'stable_branch_probability':.5},
           'payment_and_counts': arithmetic, 'allocations': all_results,
           'candidate_selected_for_null_check': best, 'candidate_null': null,
           'payment_caveat': 'Model-based complete-choice expected payments. Observation MCAR does not specify withdrawal times or justify cancelling accrued rewards; no dropout payment saving is assumed.'}
    (OUT/'comparison_results.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'completed': str(OUT/'comparison_results.json'), 'best_candidate': best}), flush=True)


if __name__ == '__main__':
    main()
