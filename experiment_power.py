#!/usr/bin/env python3
"""Final prospective power: joint signs primary, nonuniform iid friction."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE
ROOT_DIR = HERE
SOURCE = ROOT_DIR/'power_helpers.py'
spec = importlib.util.spec_from_file_location('final_joint_frozen_power', SOURCE)
power = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = power
spec.loader.exec_module(power)

PROBABILITIES = (.4, .2, .4)
EFFECT_NAMES = ('positive_endpoint_interaction', 'near_depth_4_minus_0',
                'far_depth_4_minus_0', 'depth_far_minus_near_gamma0',
                'conversion_far_minus_near_gamma0',
                'conversion_far_minus_near_gamma2',
                'conversion_far_minus_near_gamma4',
                'CS_near_minus_far_positive_gamma')
SECONDARY_NAMES = ('positive_endpoint_interaction',
                   'gamma0_depth_ranking_equivalence',
                   'conversion_ranking_equivalence_all_gamma',
                   'near_CS_advantage_positive_gamma')
OUTCOME_NAMES = ('T', 'purchase', 'induced_CS')
CONFIG = power.reference.Config(rounds=24, pairs=150,
                                gamma_probabilities=PROBABILITIES)


def draw_design(rng, batch):
    c = rng.integers(0, 2, (batch, power.P, power.J))
    g = rng.choice(3, size=(batch, power.P, power.J), p=PROBABILITIES)
    u = rng.integers(4900, 5101, (batch, power.P, 1))
    prices = rng.integers(2000, 12001, (batch, power.P, power.J, 4))
    return c, g, u, prices


def behavioural_outcomes(rng, design, alpha, concentration, stable_share, cached):
    """Attentive, fixed-depth and zero-friction policies."""
    c, g, u, prices = design
    batch = len(c)
    if alpha in (0., 1.):
        attention = np.full((batch, power.P, 2, 1), alpha)
    else:
        attention = rng.beta(alpha*concentration, (1-alpha)*concentration,
                             (batch, power.P, 2, 1))
    stable_k = rng.integers(0, 5, (batch, power.P, 2, 1))
    rational = rng.random((batch, power.P, 2, power.J)) < attention
    stable_branch = rng.random((batch, power.P, 2, power.J)) < stable_share
    outcomes = np.empty((3, batch, power.P, 2, power.J))
    for a in range(2):
        depth_opt, depth_g0 = cached[a]
        depth = np.where(rational[:, :, a], depth_opt,
                         np.where(stable_branch[:, :, a], stable_k[:, :, a], depth_g0))
        conversion, cs = power.account_depth(prices, u, c, g, a, depth)
        outcomes[0, :, :, a] = depth
        outcomes[1, :, :, a] = conversion
        outcomes[2, :, :, a] = cs
    return outcomes


def reclassify(old):
    """Only the confirmatory hierarchy changes; component tests are unchanged."""
    secondary_raw = np.column_stack((old['primary_p'], old['raw_family_p'][:,1:]))
    order = np.argsort(secondary_raw, axis=1, kind='stable')
    sorted_p = np.take_along_axis(secondary_raw, order, axis=1)
    sorted_adjusted = np.minimum(1, np.maximum.accumulate(sorted_p*np.array([4,3,2,1]), axis=1))
    adjusted = np.empty_like(sorted_adjusted)
    np.put_along_axis(adjusted, order, sorted_adjusted, axis=1)
    return {'estimate': old['estimate'], 'se': old['se'], 'df': old['df'],
            'valid': old['valid'], 'primary_joint_sign_p': old['raw_family_p'][:,0],
            'secondary_raw_p': secondary_raw, 'secondary_holm_p': adjusted}


def observed_cell_means(outcomes, geometry):
    """Cell means with equal period weights."""
    x, counts, observed = (geometry[k] for k in ('x','n','observed'))
    batch = len(x)
    output = np.empty((batch, 3, 2, 2, 3))
    for k, yy in enumerate(outcomes):
        mu = np.zeros((batch, 2, power.L))
        for a in range(2):
            np.add.at(mu[:,a], (np.arange(batch)[:,None,None], x),
                      yy[:,:,a]*observed[:,:,a,None])
        mu = np.divide(mu, counts, out=np.full_like(mu, np.nan), where=counts>0)
        output[:,k] = mu.reshape(batch,2,2,3,power.J).mean(axis=-1)
    return output


def validate(seed):
    rng = np.random.default_rng(seed)
    design = draw_design(rng, 3)
    c,g,u,prices = design
    cached = [(power.policy_depth(prices,u,c,g,a),
               power.policy_depth(prices,u,c,np.zeros_like(g),a)) for a in range(2)]
    for alpha in (0.,.5,.75,1.):
        mine = behavioural_outcomes(np.random.default_rng(seed+1), design, alpha,5,.5,cached)
        original = power.behavioural_outcomes(np.random.default_rng(seed+1), design,alpha,cached)
        assert np.array_equal(mine, original)
    outcomes = behavioural_outcomes(rng,design,.5,5,.5,cached)
    observed = rng.random((3,power.P,2))>=.1
    geometry = power.geometry(c,g,observed)
    result = reclassify(power.analyse_batch(outcomes,geometry))
    full_means = observed_cell_means(outcomes,geometry)
    # Independent aggregate linear contrasts recover every output effect.
    from_means = np.empty((3,8))
    for q in range(8):
        from_means[:,q] = np.sum(full_means[:,power.OUTCOME_INDEX[q]]
                                *power.COEF[q][None,:,None,:]*power.J,axis=(1,2,3))
    assert np.allclose(from_means,result['estimate'],atol=1e-12)
    errors=[]; decisions=0
    for trial in range(3):
        data = power.reference_data(design,outcomes,observed,trial)
        rr=[]
        kinds=[('T','interaction',2),('T','near_simple',2),('T','far_simple',2),
               ('T','rank_at_gamma',0),('purchase','rank_at_gamma',0),
               ('purchase','rank_at_gamma',1),('purchase','rank_at_gamma',2),
               ('induced_CS','near_CS_positive_gamma',2)]
        for q,(outcome,kind,gamma) in enumerate(kinds):
            reference=power.reference.cr2(data,outcome,power.reference.contrast(power.J,kind,gamma),CONFIG)
            rr.append(reference)
            for field in ('estimate','se','df'):
                errors.append(abs(reference[field]-result[field][trial,q]))
                assert np.isclose(reference[field],result[field][trial,q],rtol=1e-10,atol=1e-10)
        primary=power.reference.joint_signs_primary(rr[1],rr[2],.05)
        assert np.isclose(primary['primary_p'],result['primary_joint_sign_p'][trial],atol=1e-12)
        raw=[rr[0]['p_greater'],power.reference.tost(rr[3],.25)['p'],
             power.reference.joint_max([power.reference.tost(rr[q],.05)['p'] for q in (4,5,6)]),
             rr[7]['p_greater']]
        adjusted=power.reference.holm(dict(zip(SECONDARY_NAMES,raw)))
        assert np.allclose(raw,result['secondary_raw_p'][trial],atol=1e-12)
        assert np.allclose([adjusted[k]['holm_p'] for k in SECONDARY_NAMES],
                           result['secondary_holm_p'][trial],atol=1e-12)
        decisions+=1
    return {'base_DGP_array_identical_to_frozen_helper':True,
            'reference_analysis_version':power.reference.ANALYSIS_VERSION,
            'numerical_estimate_se_df_checks':len(errors),
            'max_absolute_numeric_discrepancy':max(errors),
            'new_primary_and_secondary_family_reference_trials':decisions,
            'cell_mean_contrast_identities_verified':True}


def rates(events):
    events=np.asarray(events,bool); n=len(events); p=events.mean(axis=0)
    return {'rate':p.tolist(),'successes':events.sum(axis=0).tolist(),
            'Monte_Carlo_SE':np.sqrt(p*(1-p)/n).tolist()}


def summarize(arrays, alpha, concentration, stable_share):
    n=len(arrays['estimate']); cell=arrays['cell_means']
    cell_n=np.isfinite(cell).sum(axis=0)
    cell_mu=np.nanmean(cell,axis=0)
    cell_se=np.nanstd(cell,axis=0,ddof=1)/np.sqrt(cell_n)
    # A helper may retain arithmetic placeholders after a sparse-cell failure.
    # They never count as a usable estimate or uncertainty measure in summaries.
    valid=arrays['valid']; effect_n=valid.sum(axis=0)
    finite_effect=np.where(valid,arrays['estimate'],np.nan)
    finite_se=np.where(valid,arrays['se'],np.nan)
    finite_df=np.where(valid,arrays['df'],np.nan)
    return {'replications':n,'alpha_assumed':alpha,'concentration':concentration,'stable_share':stable_share,
            'primary_joint_signs':rates(arrays['primary_joint_sign_p']<.05),
            'primary_inestimability':rates(~np.all(arrays['valid'][:,1:3],axis=1)),
            'secondary_names':SECONDARY_NAMES,
            'secondary_unadjusted':rates(arrays['secondary_raw_p']<.05),
            'secondary_Holm_adjusted':rates(arrays['secondary_holm_p']<.05),
            'secondary_inestimability':rates(np.column_stack((~arrays['valid'][:,0],~arrays['valid'][:,3],
                                                        ~np.all(arrays['valid'][:,4:7],axis=1),~arrays['valid'][:,7]))),
            'effect_names':EFFECT_NAMES,
            'effect_mean':np.nanmean(finite_effect,axis=0).tolist(),
            'effect_mean_Monte_Carlo_SE':(np.nanstd(finite_effect,axis=0,ddof=1)/np.sqrt(effect_n)).tolist(),
            'mean_estimated_SE':np.nanmean(finite_se,axis=0).tolist(),
            'empirical_estimator_SD':np.nanstd(finite_effect,axis=0,ddof=1).tolist(),
            'mean_df':np.nanmean(finite_df,axis=0).tolist(),
            'effect_estimable_replications':effect_n.tolist(),
            'effect_summary_conditioning':'Means, standard errors and degrees of freedom average only inference-estimable replications; test rejection rates retain all replications.',
            'contrast_inestimability':rates(~arrays['valid']),
            'cell_mean_axes':['outcome: T,purchase,induced_CS','ranking: nearest,farthest','fee: 1,4','gamma: 0,2,4'],
            'cell_mean_over_estimable_simulations':cell_mu.tolist(),
            'cell_mean_Monte_Carlo_SE':cell_se.tolist(),
            'cell_mean_estimable_replications':cell_n.tolist()}


def run_group(name, alphas, concentration, stable_share, repetitions, batch_size, seed, output_dir):
    rng=np.random.default_rng(seed); start=time.time(); pieces={a:[] for a in alphas}; sparse=[]
    for offset in range(0,repetitions,batch_size):
        batch=min(batch_size,repetitions-offset)
        design=draw_design(rng,batch); c,g,u,prices=design
        cached=[(power.policy_depth(prices,u,c,g,a),
                 power.policy_depth(prices,u,c,np.zeros_like(g),a)) for a in range(2)]
        observed=rng.random((batch,power.P,2))>=.1
        geometry=power.geometry(c,g,observed)
        sparse.append(np.any(geometry['n']<2,axis=(1,2)))
        for alpha in alphas:
            outcomes=behavioural_outcomes(rng,design,alpha,concentration,stable_share,cached)
            result=reclassify(power.analyse_batch(outcomes,geometry))
            result['cell_means']=observed_cell_means(outcomes,geometry)
            pieces[alpha].append(result)
        if offset%(batch_size*10)==0:
            print(json.dumps({'group':name,'completed':offset+batch,'target':repetitions,
                              'elapsed_seconds':round(time.time()-start,2)}),flush=True)
    group={'seed':seed,'concentration':concentration,'stable_share':stable_share,
           'replications_per_alpha':repetitions,'runtime_seconds':time.time()-start,
           'any_full_cell_sparse':rates(np.concatenate(sparse)),'scenarios':{}}
    all_arrays={}
    for alpha in alphas:
        arrays={k:np.concatenate([p[k] for p in pieces[alpha]]) for k in pieces[alpha][0]}
        label=f'alpha_{alpha:g}'
        group['scenarios'][label]=summarize(arrays,alpha,concentration,stable_share)
        for key,value in arrays.items():all_arrays[f'{label}__{key}']=value
    (output_dir/f'{name}_results.json').write_text(json.dumps(group,indent=2)+'\n')
    np.savez_compressed(output_dir/f'{name}_SIMULATION_SUMMARIES.npz',**all_arrays)
    print(json.dumps({'group_complete':name,'runtime_seconds':group['runtime_seconds']}),flush=True)
    return group


def source_hashes():
    paths=[SOURCE,ROOT_DIR/'experiment_analysis.py',
           ROOT_DIR/'experiment_model.py',
           ROOT_DIR/'results/calibration/policy_thresholds.npz']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--replications',type=int,default=10000)
    ap.add_argument('--strong-replications',type=int,default=10000)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--base-seed',type=int,default=1710)
    ap.add_argument('--strong-seed',type=int,default=1710)
    ap.add_argument('--output-dir',type=Path,default=HERE/'results/power')
    ap.add_argument('--validation-only',action='store_true')
    args=ap.parse_args()
    if min(args.replications,args.strong_replications,args.batch)<2:
        ap.error('Replication counts and batch size must be at least two')
    before=source_hashes(); validation=validate(2026091420)
    print(json.dumps({'validation':validation}),flush=True)
    if args.validation_only:return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    base=run_group('base',(1.,.75,.5,.25,0.),5.,.5,args.replications,args.batch,args.base_seed,args.output_dir)
    strong=run_group('strong_persistence',(.5,.25,0.),.5,1.,args.strong_replications,args.batch,args.strong_seed,args.output_dir)
    after=source_hashes()
    output={'status':'PROSPECTIVE ASSUMED-DGP POWER; NO PARTICIPANT OR PILOT DATA',
            'simulation_version':'final-joint-primary-nonuniform-1.0',
            'runtime_versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
            'design':{'participants':300,'pairs':150,'rounds':24,'fees':[1,4],'fee_probabilities':[.5,.5],
                      'gammas':[0,2,4],'gamma_probabilities':PROBABILITIES,'whole_person_MCAR':.1,
                      'primary':'near endpoint <0 AND far endpoint >0, maximum one-sided component p, no Holm',
                      'secondary_fixed_Holm_family':SECONDARY_NAMES},
            'validation':validation,'source_hashes_before':before,'source_hashes_after':after,
            'source_hashes_unchanged':before==after,'base':base,'strong_persistence':strong,
            'scope':['Alpha is assumed responsiveness, not an estimate or pilot calibration.',
                     'The both-zero alpha=0 DGP is not the least-favourable union null.',
                     'Inestimable tests remain in the simulation denominator as non-rejections.',
                     'Observation loss is MCAR in these final planning scenarios, not inferred to be realistic.',
                     'Numerical counterfactual outcomes are not empirical participant observations.']}
    (args.output_dir/'final_power_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'complete':str(args.output_dir/'final_power_results.json')}),flush=True)


if __name__=='__main__':main()
