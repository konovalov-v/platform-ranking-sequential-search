"""Endpoint40 allocation: fabricated/model-only planning, never real payments."""

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / ""
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("budget_endpoint40_finite_model", MODEL_DIR / "experiment_model.py")
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)
CFG = model.ModelConfig.load()
BANK_PATH = MODEL_DIR / "results/calibration/policy_thresholds.npz"
BANK = model.PolicyBank.load(BANK_PATH, CFG)
J, STUDY_PAIRS, BASE_CENTS, PRIZE_CENTS, M_H = 24, 150, 300, 1200, 20000
GAMMA_PROBABILITIES = np.array([.4, .2, .4])
EXACT_CELLS_PATH = HERE / "results/exact/exact_cells.json"


def rational_scores_h(prices, u, cost, gamma_index, rank):
    """Independent vectorized implementation checked against the bank API."""
    thresholds = BANK.stop_states[rank, cost, gamma_index, np.broadcast_to(u - 4900, cost.shape)]
    best = np.zeros(cost.shape, dtype=np.int64)
    active = np.ones(cost.shape, dtype=bool)
    depth = np.zeros(cost.shape, dtype=np.int64)
    order = (0, 1, 2, 3) if rank == 0 else (3, 2, 1, 0)
    for stage, seller in enumerate(order):
        active &= best < thresholds[..., stage]
        depth += active
        incremental = 12000 - u - prices[..., seller] - 200 * gamma_index * (seller + 1)
        best = np.where(active & (incremental > best), incremental, best)
    score = 4000 + u + best - 100 * (1 + 3 * cost) * depth
    assert np.all((score >= 800) & (score <= 13900))
    return score, depth


def independent_exact_weighted_expectation():
    """Independent arbitrary-precision rational cell audit, reweighted only."""
    cells = json.loads(EXACT_CELLS_PATH.read_text())
    weights = {0: Fraction(2, 5), 2: Fraction(1, 5), 4: Fraction(2, 5)}
    expected_keys = {(r, c, g) for r in ("nearest", "farthest") for c in ("low", "high") for g in weights}
    keys = {(c["ranking"], c["cost_label"], c["gamma"]) for c in cells}
    assert len(cells) == 12 and keys == expected_keys
    mean_cs = Fraction(0)
    for cell in cells:
        value = cell["expectations"]["consumer_surplus"]
        cs = Fraction(int(value["numerator"]), int(value["denominator"]))
        mean_cs += cs * weights[cell["gamma"]] / 4
    mean_score = 40 + mean_cs
    prize_probability = mean_score / 200
    total_cash = 900 + 3600 * prize_probability
    def record(value):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator), "decimal": float(value)}
    return {"method": "Reweight independently certified rational per-cell expectations by rank1/2 * fee1/2 * gamma(2/5,1/5,2/5)",
            "weighted_mean_consumer_surplus_points": record(mean_cs),
            "weighted_mean_score_points": record(mean_score),
            "weighted_mean_prize_probability": record(prize_probability),
            "expected_total_euros": record(total_cash)}


def exact_type_means():
    """Finite expectation arithmetic, not sampled prices: E[z_h | rank,u]."""
    solver = model.FiniteSolver(CFG)
    means = np.zeros((2, 201))
    checked = 0
    for rank, rank_name in enumerate(CFG.ranking_names):
        for ci, cost in enumerate(CFG.cost_names):
            for gi, gamma in enumerate(CFG.data["gammas"]):
                for ui, u in enumerate(CFG.outside_units):
                    policy = solver.solve_policy(rank_name, cost, gamma, int(u))
                    assert np.array_equal(policy.stop_state_units, BANK.stop_states[rank, ci, gi, ui])
                    means[rank, ui] += (4000 + int(u) + policy.initial_value_units) * .5 * GAMMA_PROBABILITIES[gi]
                    checked += 1
    cells = json.loads((MODEL_DIR / "results/calibration/finite_cells.json").read_text())
    for rank, name in enumerate(CFG.ranking_names):
        existing = sum(.5 * GAMMA_PROBABILITIES[CFG.data["gammas"].index(c["gamma"])] * (40 + c["consumer_surplus"]) for c in cells if c["ranking"] == name)
        assert abs(means[rank].mean() / 100 - existing) < 1e-10
    expected = 2 * STUDY_PAIRS * BASE_CENTS / 100 + STUDY_PAIRS * PRIZE_CENTS / 100 * means.mean(axis=1).sum() / M_H
    independent = independent_exact_weighted_expectation()
    assert abs(expected - independent["expected_total_euros"]["decimal"]) < 1e-8
    assert abs(expected - 2853.031742089222) < 1e-8
    summary = json.loads((MODEL_DIR / "results/calibration/payment_summary.json").read_text())
    assert np.array_equal(summary["gamma_probabilities"], GAMMA_PROBABILITIES)
    assert abs(expected - 300 * summary["expected_payment_euros"]) < 1e-8
    return means, checked, independent


def pair_probabilities(mu, covariance):
    joint_win = float(mu[0] * mu[1] + covariance)
    q = np.array([1 - mu.sum() + joint_win, mu.sum() - 2 * joint_win, joint_win])
    assert np.all(q >= 0) and abs(q.sum() - 1) < 1e-12
    return q


def convolve_pairs(q, pairs=STUDY_PAIRS):
    mass = np.array([1.])
    for _ in range(pairs):
        mass = np.convolve(mass, q)
    assert np.all(mass >= 0) and abs(mass.sum() - 1) < 1e-11
    mass /= mass.sum()
    return mass


def distribution_summary(q, pairs=STUDY_PAIRS):
    mass = convolve_pairs(q, pairs)
    cash_cents = 2 * pairs * BASE_CENTS + PRIZE_CENTS * np.arange(len(mass))
    cdf = np.cumsum(mass)
    mean = float(np.dot(mass, cash_cents) / 100)
    variance = float(np.dot(mass, ((cash_cents / 100) - mean) ** 2))
    out = {"probability_total_strictly_above_3000_euros": float(mass[cash_cents > 300000].sum()),
           "quantiles_euros": {str(p): int(cash_cents[np.searchsorted(cdf, p)]) / 100 for p in (.9, .95, .99)},
           "expected_total_euros": mean, "standard_deviation_total_euros": variance ** .5}
    count_mean = q[1] + 2 * q[2]
    exact_mean = 2 * pairs * BASE_CENTS / 100 + PRIZE_CENTS / 100 * pairs * count_mean
    exact_variance = (PRIZE_CENTS / 100) ** 2 * pairs * (q[1] + 4 * q[2] - count_mean ** 2)
    assert abs(exact_mean - mean) < 1e-8
    assert abs(exact_variance - variance) < 1e-7
    return out, mass


def validate_policy(seed):
    rng = np.random.default_rng(seed)
    prices = rng.integers(2000, 12001, (201, 6, 4))
    u = np.arange(4900, 5101)[:, None]
    c = np.broadcast_to(np.repeat(np.arange(2), 3), (201, 6))
    g = np.broadcast_to(np.tile(np.arange(3), 2), (201, 6))
    checked = 0
    for rank, rank_name in enumerate(CFG.ranking_names):
        scores, depth = rational_scores_h(prices, u, c, g, rank)
        for ci, cost in enumerate(CFG.cost_names):
            for gi, gamma in enumerate(CFG.data["gammas"]):
                col = 3 * ci + gi
                reference = BANK.realised_policy(prices[:, col], u[:, 0], rank_name, cost, gamma)
                assert np.array_equal(scores[:, col], np.rint(reference["score"] * 100).astype(np.int64))
                assert np.array_equal(depth[:, col], reference["depth"])
                checked += len(prices)
    # Conditional independent lottery algebra and exact small convolution.
    q = np.array([.3, .5, .2])
    assert np.allclose(convolve_pairs(q, 2), [.09, .3, .37, .2, .04])
    return {"realised_score_and_depth_checks": checked, "convolution_two_pair_polynomial_verified": True}


def simulate_batch(seed_sequence, n, chunk_size, means_h):
    type_seed, fee_seed, gamma_seed, price_seed, bonus_seed = seed_sequence.spawn(5)
    rt, rc, rg, rp, rb = [np.random.default_rng(s) for s in (type_seed, fee_seed, gamma_seed, price_seed, bonus_seed)]
    exact_mu = means_h.mean(axis=1) / M_H
    type_covariance = float(np.mean((means_h[0] / M_H - exact_mu[0]) * (means_h[1] / M_H - exact_mu[1])))
    conditional_q_sum = np.zeros(3)
    p_sum = np.zeros(2)
    raw_product_sum, centered_pair_product_sum, centered_round_product_sum = 0., 0., 0.
    audit_wins = np.zeros(3, dtype=np.int64)
    gamma_counts = np.zeros(3, dtype=np.int64)
    done = 0
    while done < n:
        size = min(chunk_size, n - done)
        u = rt.integers(4900, 5101, (size, 1))
        c = rc.integers(0, 2, (size, J))
        g = rg.choice(3, size=(size, J), p=GAMMA_PROBABILITIES)
        gamma_counts += np.bincount(g.ravel(), minlength=3)
        prices = rp.integers(2000, 12001, (size, J, 4))
        scores = [rational_scores_h(prices, u, c, g, rank)[0] for rank in (0, 1)]
        sums_h = np.column_stack([z.sum(axis=1) for z in scores])
        probabilities = sums_h / (M_H * J)
        pn, pf = probabilities.T
        conditional_q_sum += np.array([np.sum((1 - pn) * (1 - pf)),
                                       np.sum(pn * (1 - pf) + pf * (1 - pn)), np.sum(pn * pf)])
        p_sum += probabilities.sum(axis=0)
        raw_product_sum += float(np.sum(pn * pf))
        centered_pair_product_sum += float(np.sum((pn - exact_mu[0]) * (pf - exact_mu[1])))
        centered = [(scores[r] - means_h[r, u - 4900]) / M_H for r in (0, 1)]
        centered_round_product_sum += float(np.sum(np.mean(centered[0] * centered[1], axis=1)))
        # Separate independent simulated integer prize draws are a diagnostic.
        # The main estimate integrates this avoidable Bernoulli noise exactly.
        draws = rb.integers(1, M_H * J + 1, (size, 2))
        wins = (draws <= sums_h).sum(axis=1)
        audit_wins += np.bincount(wins, minlength=3)
        done += size
    raw_q = conditional_q_sum / n
    raw_mu = p_sum / n
    direct_covariance = raw_product_sum / n - raw_mu[0] * raw_mu[1]
    pair_cv_covariance = centered_pair_product_sum / n
    # Exact identity: between-type covariance plus within-type, same-round
    # covariance divided by J. Different rounds are independent conditional u.
    within_round_covariance = centered_round_product_sum / n
    moment_covariance = type_covariance + within_round_covariance / J
    moment_q = pair_probabilities(exact_mu, moment_covariance)
    return {"simulated_pairs": n, "gamma_draw_counts": gamma_counts.tolist(), "raw_conditional_pair_probabilities": raw_q.tolist(),
            "raw_marginal_prize_probabilities": raw_mu.tolist(), "raw_pair_win_covariance": direct_covariance,
            "pair_centered_control_variate_covariance": pair_cv_covariance,
            "exact_between_type_covariance": type_covariance,
            "same_round_centered_covariance_estimate": within_round_covariance,
            "moment_pair_win_covariance": moment_covariance,
            "moment_pair_probabilities": moment_q.tolist(),
            "independent_integer_draw_win_frequencies": (audit_wins / n).tolist(),
            "raw_distribution": distribution_summary(raw_q)[0],
            "moment_distribution": distribution_summary(moment_q)[0]}


OUT = HERE / "results/budget"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1710)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--pairs-per-batch", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()
    if args.batches < 2 or args.pairs_per_batch < 1 or args.chunk_size < 1:
        parser.error("need at least two positive batches and a positive chunk size")
    started = time.monotonic()
    checks = validate_policy(args.seed + 1)
    means_h, checked, independent_exact = exact_type_means()
    checks["solved_policies_checked_against_frozen_bank"] = checked
    checks["weighted_policy_mean_matches_independent_rational_cell_audit"] = True
    checks["current_endpoint40_payment_summary_crosscheck"] = True
    exact_mu = means_h.mean(axis=1) / M_H
    print(f"Exact finite marginal win probabilities: {exact_mu.tolist()}", flush=True)
    batches = []
    for i, seed in enumerate(np.random.SeedSequence(args.seed).spawn(args.batches)):
        result = simulate_batch(seed, args.pairs_per_batch, args.chunk_size, means_h)
        batches.append(result)
        print(f"Batch {i+1}/{args.batches}: {args.pairs_per_batch} pairs; P(cash>3000) raw={result['raw_distribution']['probability_total_strictly_above_3000_euros']:.8f}, moment={result['moment_distribution']['probability_total_strictly_above_3000_euros']:.8f}", flush=True)
    raw_q = np.mean([b["raw_conditional_pair_probabilities"] for b in batches], axis=0)
    moment_q = np.mean([b["moment_pair_probabilities"] for b in batches], axis=0)
    raw_summary, _ = distribution_summary(raw_q)
    summary, mass = distribution_summary(moment_q)
    independent_q = pair_probabilities(exact_mu, 0.)
    independent_summary, _ = distribution_summary(independent_q)
    def uncertainty(path):
        def extract(batch):
            out = batch
            for key in path:
                out = out[key]
            return out
        values = np.array([extract(b) for b in batches])
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        center = float(values.mean())
        width = float(student_t.ppf(.975, len(values)-1) * se)
        return {"batch_estimates": values.tolist(), "mean_batch_estimate": center,
                "Monte_Carlo_standard_error_of_pooled_estimate": se,
                "approximate_95_percent_MC_interval": [center-width, center+width],
                "method": f"independent equal-size batches; t{len(values)-1} interval for{len(values)}batches; numerical MC uncertainty only"}
    cov = float(np.mean([b["moment_pair_win_covariance"] for b in batches]))
    exact_expected = 2 * STUDY_PAIRS * BASE_CENTS / 100 + STUDY_PAIRS * PRIZE_CENTS / 100 * exact_mu.sum()
    report = {"label": "Model-based participant payments",
              "allocation": {"gamma_values": [0, 2, 4], "gamma_probabilities": GAMMA_PROBABILITIES.tolist(),
                             "fee_probabilities_low_high": [.5, .5], "current_conditions": "iid pair-common each round",
                             "status": "prespecified"},
              "independent_exact_weighted_validation": independent_exact,
              "gamma_draw_counts": np.sum([b["gamma_draw_counts"] for b in batches], axis=0).tolist(),
              "seed": args.seed, "simulated_pairs": args.batches * args.pairs_per_batch,
              "independent_batches": args.batches, "pairs_per_batch": args.pairs_per_batch,
              "chunk_size": args.chunk_size, "actual_study_participants": 300, "actual_study_pairs": 150,
              "planned_rounds_per_participant": J, "fixed_fee_euros": 3, "binary_prize_euros": 12,
              "contractual_maximum_euros": 4500, "strict_3000_threshold_requires_at_least_wins": 176,
              "assumptions": ["all 300 participants enroll and finish 24 rounds", "optimal finite-grid policies and specified ties",
                              "pair-shared u0 drawn once, iid pair-shared current fee/gamma and seller vectors each round",
                              "independent pairs; prizes conditionally independent across people given their scores",
                              "no processing, platform, screening, pilot or infrastructure cost in these participant-cash totals",
                              "no real participant data and no actual payments; selected gamma allocation now (0.4,0.2,0.4); fees/prize/N/J unchanged"],
              "sources_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), EXACT_CELLS_PATH,
                                           MODEL_DIR / "experiment_model.py", MODEL_DIR / "experiment_config.json", BANK_PATH,
                                           MODEL_DIR / "results/calibration/finite_cells.json", MODEL_DIR / "results/calibration/payment_summary.json")},
              "checks": checks, "exact_marginal_prize_probabilities_near_far": exact_mu.tolist(),
              "exact_expected_total_euros": float(exact_expected),
              "screening": {"assumed_pass_rate": .8, "expected_screens": 300/.8,
                            "expected_failures": 300/.8-300, "payment_per_failure_euros": 1,
                            "expected_screening_euros": (300/.8-300)*1,
                            "expected_total_including_screening_euros": float(exact_expected)+(300/.8-300)*1},
              "direct_conditional_average_pair_probabilities_0_1_2_wins": raw_q.tolist(),
              "direct_conditional_average_distribution": raw_summary,
              "moment_variance_reduced_pair_probabilities_0_1_2_wins": moment_q.tolist(),
              "moment_variance_reduced_distribution": summary,
              "moment_pair_win_covariance": cov,
              "pair_win_correlation": float(cov / np.sqrt(np.prod(exact_mu * (1-exact_mu)))),
              "independent_participant_counterfactual_distribution": independent_summary,
              "Monte_Carlo_sensitivity": {
                  "raw_expected_total_euros": uncertainty(("raw_distribution", "expected_total_euros")),
                  "raw_probability_above_3000": uncertainty(("raw_distribution", "probability_total_strictly_above_3000_euros")),
                  "moment_probability_above_3000": uncertainty(("moment_distribution", "probability_total_strictly_above_3000_euros")),
                  "moment_pair_win_covariance": uncertainty(("moment_pair_win_covariance",)),
                  "batch_quantiles_euros": [b["moment_distribution"]["quantiles_euros"] for b in batches]},
              "independent_integer_draw_win_frequencies": np.mean([b["independent_integer_draw_win_frequencies"] for b in batches], axis=0).tolist(),
              "batches": batches, "elapsed_seconds": time.monotonic()-started}
    (OUT / "budget_results.json").write_text(json.dumps(report, indent=2)+"\n")
    with (OUT / "payout_distribution.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["total_bonus_wins", "participant_cash_euros", "probability", "cumulative_probability"])
        cumulative = np.cumsum(mass)
        for k, probability in enumerate(mass):
            writer.writerow([k, 900+12*k, format(probability, ".17g"), format(cumulative[k], ".17g")])
    print(json.dumps({"exact_expected_total_euros": report["exact_expected_total_euros"],
                      "direct": raw_summary, "moment": summary, "pair_covariance": cov,
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
