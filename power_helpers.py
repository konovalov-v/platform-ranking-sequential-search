"""Batched cell means, inference and behavioural simulation helpers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model = import_path("secondary_finite_model", ROOT_DIR / "experiment_model.py")
reference = import_path("secondary_reference_analysis", ROOT_DIR / "experiment_analysis.py")
CFG = model.ModelConfig.load()
BANK = model.PolicyBank.load(ROOT_DIR / "results/calibration/policy_thresholds.npz", CFG)
P, J, L = 150, 24, 144
ALPHAS = (1., .75, .5, 0.)
EFFECT_NAMES = ("primary_depth_interaction", "near_depth_4_minus_0", "far_depth_4_minus_0",
                "depth_far_minus_near_gamma0", "conversion_far_minus_near_gamma0",
                "conversion_far_minus_near_gamma2", "conversion_far_minus_near_gamma4",
                "CS_near_minus_far_positive_gamma")
FAMILY_NAMES = ("sign_reversal_IUT", "depth_gamma0_equivalence", "conversion_all_gamma_equivalence", "near_CS_advantage")
OUTCOME_INDEX = (0, 0, 0, 0, 1, 1, 1, 2)
COEF = np.zeros((8, 2, 3))
COEF[0, 0, [0, 2]] = [1, -1]
COEF[0, 1, [0, 2]] = [-1, 1]
COEF[1, 0, [0, 2]] = [-1, 1]
COEF[2, 1, [0, 2]] = [-1, 1]
COEF[3:5, 0, 0] = -1
COEF[3:5, 1, 0] = 1
COEF[5, :, 1] = [-1, 1]
COEF[6, :, 2] = [-1, 1]
COEF[7, 0, 1:] = .5
COEF[7, 1, 1:] = -.5
COEF /= 2*J


def draw_design(rng, batch):
    c = rng.integers(0, 2, (batch, P, J))
    g = rng.integers(0, 3, (batch, P, J))
    u = rng.integers(4900, 5101, (batch, P, 1))
    prices = rng.integers(2000, 12001, (batch, P, J, 4))
    return c, g, u, prices


def policy_depth(prices, u, c, g, rank):
    """Vectorised exact bank policy, with seller-indexed prices and integer state."""
    thresholds = BANK.stop_states[rank, c, g, np.broadcast_to(u-4900, c.shape)]
    best = np.zeros(c.shape, dtype=np.int64)
    active = np.ones(c.shape, dtype=bool)
    depth = np.zeros(c.shape, dtype=np.int64)
    order = (0, 1, 2, 3) if rank == 0 else (3, 2, 1, 0)
    for stage, seller in enumerate(order):
        active &= best < thresholds[..., stage]
        depth += active
        reward = 12000-u-prices[..., seller]-200*g*(seller+1)
        best = np.where(active & (reward > best), reward, best)
    return depth


def account_depth(prices, u, c, g, rank, depth):
    best = np.zeros(c.shape, dtype=np.int64)
    order = (0, 1, 2, 3) if rank == 0 else (3, 2, 1, 0)
    for stage, seller in enumerate(order):
        reward = 12000-u-prices[..., seller]-200*g*(seller+1)
        best = np.where((depth > stage) & (reward > best), reward, best)
    purchase = best > 0
    cs = (u+best)/100 - depth*(1+3*c)
    return purchase, cs


def behavioural_outcomes(rng, design, alpha, cached_depths=None):
    c, g, u, prices = design
    batch = len(c)
    if cached_depths is None:
        cached_depths = [(policy_depth(prices, u, c, g, a),
                          policy_depth(prices, u, c, np.zeros_like(g), a)) for a in range(2)]
    if alpha in (0., 1.):
        attention = np.full((batch, P, 2, 1), alpha)
    else:
        attention = rng.beta(alpha*5, (1-alpha)*5, (batch, P, 2, 1))
    stable_k = rng.integers(0, 5, (batch, P, 2, 1))
    rational = rng.random((batch, P, 2, J)) < attention
    stable_branch = rng.random((batch, P, 2, J)) < .5
    ys = np.empty((3, batch, P, 2, J))
    for a in range(2):
        depth_opt, depth_g0 = cached_depths[a]
        depth = np.where(rational[:, :, a], depth_opt,
                         np.where(stable_branch[:, :, a], stable_k[:, :, a], depth_g0))
        conversion, cs = account_depth(prices, u, c, g, a, depth)
        ys[0, :, :, a] = depth
        ys[1, :, :, a] = conversion
        ys[2, :, :, a] = cs
    return ys


def geometry(c, g, observed):
    """Exact reference CR2 geometry, shared by outcomes with the same mask."""
    batch = len(c)
    x = (c*3+g)*J+np.arange(J)
    z = np.zeros((batch, P, L))
    np.put_along_axis(z, x, 1., axis=2)
    n = np.einsum("bpl,bpa->bal", z, observed)
    denom = np.sqrt(np.maximum(n*(n-1), 1))
    ell = np.where(n > 1, 1/denom, 0.)
    base = np.empty((batch, 2, 3, P, P))
    ix = np.arange(P)
    cell_gamma = (np.arange(L)//J) % 3
    for a in range(2):
        za = z * observed[:, :, a, None]
        for gg in range(3):
            mask = cell_gamma == gg
            zz = za[:, :, mask]
            ll = ell[:, a, mask]
            nn = n[:, a, mask]
            vv = zz * (ll / np.sqrt(np.maximum(nn, 1)))[:, None, :]
            G = -np.einsum("bpl,bql->bpq", vv, vv, optimize=False)
            diagonal = np.einsum("bpl,bl->bp", zz, ll**2)
            G[:, ix, ix] += diagonal
            base[:, a, gg] = G
    dfs = np.empty((batch, 8))
    valid = np.empty((batch, 8), bool)
    for q in range(8):
        support = COEF[q] != 0
        combined = base[:, support].sum(axis=1)
        tr = np.trace(combined, axis1=-2, axis2=-1)
        norm2 = np.sum(combined**2, axis=(-1, -2))
        dfs[:, q] = np.divide(tr**2, norm2, out=np.ones(batch), where=norm2 > 0)
        valid[:, q] = np.all(n[:, np.repeat(support[:, None, :], 2, axis=1).reshape(2, 6).repeat(J, axis=1)] > 1, axis=1)
    return {"x": x, "z": z, "n": n, "ell": ell, "df": dfs, "valid": valid,
            "observed": observed, "g": g}


def analyse_batch(ys, geom):
    batch = ys.shape[1]
    x, z, n, ell, g, observed = (geom[k] for k in ("x", "z", "n", "ell", "g", "observed"))
    estimates = np.empty((batch, 8))
    se = np.empty_like(estimates)
    residuals = []
    means = []
    for yy in ys:
        # There is exactly one observation at each pair/period. Scatter sums.
        mu = np.zeros((batch, 2, L))
        for a in range(2):
            np.add.at(mu[:, a], (np.arange(batch)[:, None, None], x), yy[:, :, a]*observed[:, :, a, None])
        mu = np.divide(mu, n, out=np.zeros_like(mu), where=n > 0)
        means.append(mu)
        prediction = np.stack([np.take_along_axis(mu[:, a, None, :], x, axis=2) for a in range(2)], axis=2)
        residuals.append((yy-prediction)*observed[..., None])
    for q in range(8):
        mu = means[OUTCOME_INDEX[q]].reshape(batch, 2, 2, 3, J)
        estimates[:, q] = np.sum(mu*COEF[q][None, :, None, :, None], axis=(1, 2, 3, 4))
        scores = np.zeros((batch, P))
        residual = residuals[OUTCOME_INDEX[q]]
        for a in range(2):
            local_ell = np.take_along_axis(ell[:, a, None, :], x, axis=2)
            scores += np.sum(residual[:, :, a]*local_ell*COEF[q, a][g], axis=2)
        se[:, q] = np.sqrt(np.sum(scores**2, axis=1))
    observed_scale = np.max(np.abs(ys)*observed[None, ..., None], axis=(2, 3, 4))
    numerical_zero_tolerance = 1e-12*np.maximum(1., observed_scale[np.array(OUTCOME_INDEX)].T)
    valid = geom["valid"] & (se > numerical_zero_tolerance)
    df = geom["df"]
    tt = np.divide(estimates, se, out=np.zeros_like(estimates), where=se > 0)
    greater, less = student_t.sf(tt, df), student_t.cdf(tt, df)
    greater[~valid] = 1
    less[~valid] = 1
    margin = np.array([.25, .05, .05, .05])
    eq = student_t.sf((margin-abs(estimates[:, 3:7]))/np.maximum(se[:, 3:7], 1e-100), df[:, 3:7])
    eq[~valid[:, 3:7]] = 1
    raw = np.column_stack([np.maximum(less[:, 1], greater[:, 2]), eq[:, 0], eq[:, 1:].max(axis=1), greater[:, 7]])
    order = np.argsort(raw, axis=1, kind="stable")
    adjusted_ordered = np.minimum(1, np.maximum.accumulate(np.take_along_axis(raw, order, axis=1)*np.array([4, 3, 2, 1]), axis=1))
    adjusted = np.empty_like(raw)
    np.put_along_axis(adjusted, order, adjusted_ordered, axis=1)
    return {"estimate": estimates, "se": se, "df": df, "valid": valid,
            "raw_family_p": raw, "holm_family_p": adjusted, "primary_p": greater[:, 0]}


def reference_data(design, ys, observed, trial):
    c, g, *_ = design
    return {"pair": np.broadcast_to(np.arange(P)[:, None, None], (P, 2, J)).ravel(),
            "rank": np.broadcast_to(np.arange(2)[None, :, None], (P, 2, J)).ravel(),
            "round": np.broadcast_to(np.arange(J), (P, 2, J)).ravel(),
            "cost": np.broadcast_to(c[trial, :, None], (P, 2, J)).ravel(),
            "gamma": np.broadcast_to(g[trial, :, None], (P, 2, J)).ravel(),
            **{name: np.where(observed[trial, :, :, None], ys[oi, trial], np.nan).ravel()
               for oi, name in enumerate(("T", "purchase", "induced_CS"))}}


def validate(rng):
    design = draw_design(rng, 3)
    ys = behavioural_outcomes(rng, design, .5)
    errors = []
    checks = 0
    for loss in (0., .1):
        observed = rng.random((3, P, 2)) >= loss
        geom = geometry(design[0], design[1], observed)
        got = analyse_batch(ys, geom)
        for b in range(3):
            data = reference_data(design, ys, observed, b)
            weights = [reference.contrast(J, "interaction"), reference.contrast(J, "near_simple"),
                       reference.contrast(J, "far_simple"), reference.contrast(J, "rank_at_gamma", 0),
                       reference.contrast(J, "rank_at_gamma", 0), reference.contrast(J, "rank_at_gamma", 1),
                       reference.contrast(J, "rank_at_gamma", 2), reference.contrast(J, "near_CS_positive_gamma")]
            rr = []
            for q, w in enumerate(weights):
                r = reference.cr2(data, ("T", "purchase", "induced_CS")[OUTCOME_INDEX[q]], w, reference.Config())
                for key in ("estimate", "se", "df"):
                    error = abs(r[key]-got[key][b, q])
                    errors.append(error)
                    assert np.isclose(r[key], got[key][b, q], rtol=1e-10, atol=1e-10), (q, key, r[key], got[key][b, q])
                    checks += 1
                rr.append(r)
            raw = [max(rr[1]["p_less"], rr[2]["p_greater"]), reference.tost(rr[3], .25)["p"],
                   max(reference.tost(rr[q], .05)["p"] for q in (4, 5, 6)), rr[7]["p_greater"]]
            h = reference.holm(dict(zip(FAMILY_NAMES, raw)))
            assert np.allclose(raw, got["raw_family_p"][b], atol=1e-12)
            assert np.allclose([h[n]["holm_p"] for n in FAMILY_NAMES], got["holm_family_p"][b], atol=1e-12)
    # Check own vectorised policy/accounting against public bank API.
    c, g, u, prices = design
    policy_checks = 0
    for a, rank in enumerate(CFG.ranking_names):
        depths = policy_depth(prices, u, c, g, a)
        buy, cs = account_depth(prices, u, c, g, a, depths)
        ub = np.broadcast_to(u, c.shape)
        for ci, cost in enumerate(CFG.cost_names):
            for gi, gamma in enumerate(CFG.data["gammas"]):
                select = (c == ci) & (g == gi)
                actual = BANK.realised_policy(prices[select], ub[select], rank, cost, gamma)
                for key, mine in (("depth", depths), ("conversion", buy), ("consumer_surplus", cs)):
                    assert np.allclose(actual[key], mine[select], atol=1e-12)
                    policy_checks += int(select.sum())
    return {"reference_trials": 6, "estimate_se_df_comparisons": checks,
            "max_absolute_numeric_error": max(errors), "family_raw_and_Holm_verified": True,
            "bank_outcome_comparisons": policy_checks}


def summarize(results):
    est = results["estimate"]
    reps = len(est)
    def probability(x):
        p = x.mean(axis=0)
        return {"rate": p.tolist(), "Monte_Carlo_SE": np.sqrt(p*(1-p)/reps).tolist(),
                "successes": x.sum(axis=0).tolist()}
    return {"replications": reps, "effect_names": EFFECT_NAMES,
            "estimated_effect_mean": est.mean(axis=0).tolist(),
            "effect_mean_Monte_Carlo_SE": (est.std(axis=0, ddof=1)/np.sqrt(reps)).tolist(),
            "mean_estimated_SE": results["se"].mean(axis=0).tolist(),
            "empirical_estimator_SD": est.std(axis=0, ddof=1).tolist(),
            "mean_df": results["df"].mean(axis=0).tolist(),
            "nonestimable_trials_by_effect": (~results["valid"]).sum(axis=0).tolist(),
            "primary_rejection": probability(results["primary_p"] < .05),
            "secondary_family_names": FAMILY_NAMES,
            "secondary_raw_rejection": probability(results["raw_family_p"] < .05),
            "secondary_Holm_rejection": probability(results["holm_family_p"] < .05),
            "secondary_all_four_Holm_rejection": probability(np.all(results["holm_family_p"] < .05, axis=1))}


def null_outcomes(rng, c, g, mode):
    batch = len(c)
    if mode == "persistent_person_depth":
        yy = np.broadcast_to(rng.integers(0, 2, (batch, P, 2, 1))*4, (batch, P, 2, J)).copy()
    elif mode == "persistent_intercept_and_gamma_slope":
        intercept = rng.integers(0, 2, (batch, P, 2, 1))*2-1
        slope = rng.integers(0, 2, (batch, P, 2, 1))*2-1
        yy = 2+intercept+slope*(g[:, :, None]-1)
    else:
        raise ValueError(mode)
    return np.stack([yy, yy, yy]).astype(float)

