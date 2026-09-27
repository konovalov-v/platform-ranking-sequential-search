#!/usr/bin/env python3
"""Compare rational and floating-point policies and outcome means."""
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from .exact_arithmetic import HERE, ROOT, solve_integer_policy, propagate_integer_paths


def load_fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def exhaustive_toy_validation():
    profile_count = bellman_count = tie_count = 0
    configurations = []
    # Two rankings and two fees; all prices explicitly enumerated, including ties.
    for distances in ([1, 2, 3, 4], [4, 3, 2, 1]):
        for fee in (1, 2):
            L, value, outside, pmin, gamma = 13, 30, 14, 5, 1
            upper = [value - outside - pmin - gamma * d for d in distances]
            decisions, gains, cutoffs, zero_values, powers = solve_integer_policy(upper, fee, L)
            @lru_cache(None)
            def direct_value(t, s):
                if t == 4:
                    return Fraction(s)
                q = -fee + sum(direct_value(t + 1, max(s, x)) for x in range(upper[t] - L + 1, upper[t] + 1)) / L
                return max(Fraction(s), q)
            for t in range(4):
                for s in range(max(upper) + 1):
                    q = -fee + sum(direct_value(t + 1, max(s, x)) for x in range(upper[t] - L + 1, upper[t] + 1)) / L
                    assert Fraction(gains[t][s], powers[4 - t]) == q - s
                    assert decisions[t][s] == (q > s)
                    bellman_count += 1
            aggregate, depths, *_ = propagate_integer_paths(upper, distances, fee, gamma, outside, value,
                                                            L, decisions, gains, powers)
            brute = {k: 0 for k in aggregate}
            brute_depth = [0] * 5
            for prices in itertools.product(range(pmin, pmin + L), repeat=4):
                s = depth = selected_distance = 0
                for t, price in enumerate(prices):
                    if not decisions[t][s]:
                        break
                    depth += 1
                    offer = value - outside - price - gamma * distances[t]
                    tie_count += int(s > 0 and offer == s)
                    if offer > s:
                        s, selected_distance = offer, distances[t]
                purchase = int(s > 0)
                sc, tc = fee * depth, gamma * selected_distance
                rev = (value - outside) * purchase - s - tc
                cs = outside + s - sc
                raw = {"paths": 1, "depth": depth, "purchase": purchase, "surplus_units": s,
                       "distance": selected_distance, "search_cost_units": sc, "transport_cost_units": tc,
                       "revenue_units": rev, "consumer_surplus_units": cs, "welfare_units": cs + rev}
                for k, v in raw.items():
                    brute[k] += v
                brute_depth[depth] += 1
                profile_count += 1
            assert brute == aggregate
            assert brute_depth == depths
            configurations.append({"distances": distances, "fee_units": fee, "price_count": L,
                                   "thresholds": cutoffs, "profiles": L ** 4})
    # An exact indifference case verifies strict continuation, independent of full-model absence of ties.
    decisions, gains, thresholds, _, _ = solve_integer_policy([5], 1, 15)
    assert gains[0][0] == 0 and decisions[0][0] is False and thresholds[0] == 0
    return {"full_price_profiles_enumerated": profile_count, "direct_fraction_bellman_checks": bellman_count,
            "positive_incumbent_tie_encounters": tie_count, "configurations": configurations,
            "exact_zero_gain_stop_test": "passed"}


def main():
    start = time.monotonic()
    toy = exhaustive_toy_validation()
    production = HERE / ""
    # The independent implementation was completed and executed before this import was added.
    spec = importlib.util.spec_from_file_location("audited_production_model", production / "experiment_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.ModelConfig.load()
    solver = module.FiniteSolver(cfg)
    bank = module.PolicyBank.load(ROOT / "results/calibration/policy_thresholds.npz", cfg)
    exact_rows = list(csv.DictReader((ROOT / "results/exact/exact_thresholds.csv").open()))
    saved_rows = list(csv.DictReader((ROOT / "results/calibration/thresholds_by_type.csv").open()))
    key = lambda r: (r["ranking"], r["cost_label"], int(r["gamma"]), int(r["outside_units"]))
    saved_map = {key(r): r for r in saved_rows}
    certificates = {key(r): r for r in json.loads((ROOT / "results/exact/state_certificates.json").read_text())}
    policy_states = boundary_checks = 0
    maximum_boundary_gain_error_units = 0.0
    minimum_float_gain_units = float("inf")
    for row in exact_rows:
        r, c, g, u = key(row)
        expected = np.array([int(row[f"finite_stop_units_{t}"]) for t in range(4)])
        recorded = np.array([int(saved_map[(r, c, g, u)][f"finite_stop_units_{t}"]) for t in range(4)])
        assert np.array_equal(expected, recorded)
        assert np.array_equal(expected, bank.thresholds(r, c, g, np.array(u)))
        policy = solver.solve_policy(r, c, g, u)
        assert np.array_equal(expected, policy.stop_state_units)
        full_expected = np.arange(len(policy.states_units))[None, :] < expected[:, None]
        assert np.array_equal(full_expected, policy.continue_by_stage)
        # With exact audit finding no ties, every true floating sign also agrees.
        assert np.array_equal(full_expected, policy.gains_units > 0)
        policy_states += full_expected.size
        minimum_float_gain_units = min(minimum_float_gain_units, float(np.min(abs(policy.gains_units))))
        for stage in certificates[(r, c, g, u)]["stages"]:
            t, cutoff = stage["stage"], stage["stop_state_units"]
            for field, state in (("gain_at_last_continue", cutoff - 1), ("gain_at_first_stop", cutoff)):
                record = stage[field]
                if record is not None:
                    error = abs(float(load_fraction(record)) - policy.gains_units[t, state])
                    maximum_boundary_gain_error_units = max(maximum_boundary_gain_error_units, error)
                    boundary_checks += 1
    exact_cells = json.loads((ROOT / "results/exact/exact_cells.json").read_text())
    published_cells = json.loads((ROOT / "results/calibration/finite_cells.json").read_text())
    cell_key = lambda r: (r["ranking"], r["cost_label"], int(r["gamma"]))
    published_map = {cell_key(r): r for r in published_cells}
    metrics = ["depth", "conversion", "consumer_surplus", "search_cost", "transport_cost", "revenue", "welfare"]
    metric_errors, cell_reports = {}, []
    for cell in exact_cells:
        k = cell_key(cell)
        target = published_map[k]
        means = {m: load_fraction(cell["expectations"][m]) for m in metrics}
        means["chosen_distance_unconditional"] = load_fraction(cell["expectations"]["distance"])
        means["chosen_distance_given_purchase"] = load_fraction(cell["expectations"]["selected_distance_given_purchase"])
        means["price_given_purchase"] = means["revenue"] / means["conversion"]
        means["exit_probability"] = 1 - means["conversion"]
        means["entry_probability"] = 1 - load_fraction(cell["expectations"]["depth_distribution"][0])
        means["continuation_after_first"] = means["depth"] - means["entry_probability"]
        depth_distribution = list(map(load_fraction, cell["expectations"]["depth_distribution"]))
        variance = sum(t * t * p for t, p in enumerate(depth_distribution)) - means["depth"] ** 2
        values = {m: float(v) for m, v in means.items()}
        values["depth_sd"] = math.sqrt(float(variance))
        errors = {m: abs(v - target[m]) for m, v in values.items()}
        for m, error in errors.items():
            metric_errors[m] = max(metric_errors.get(m, 0), error)
        assert max(errors.values()) < 1e-9
        cell_reports.append({"ranking": k[0], "cost_label": k[1], "gamma": k[2],
                             "exact_values_as_float": values, "absolute_errors_vs_published": errors})
    exact_mean_cs = sum(load_fraction(c["expectations"]["consumer_surplus"]) for c in exact_cells) / len(exact_cells)
    exact_prize_probability = (40 + exact_mean_cs) / 200
    exact_payment = 3 + 12 * exact_prize_probability
    from .exact_arithmetic import rational_record
    report = {"toy_validation": toy, "full_policy_states_compared": policy_states,
              "type_stage_thresholds_compared": len(exact_rows) * 4,
              "threshold_mismatches": 0, "policy_mismatches": 0, "raw_float_gain_sign_mismatches": 0,
              "exact_threshold_boundary_gains_compared": boundary_checks,
              "maximum_boundary_gain_float_error_units": maximum_boundary_gain_error_units,
              "minimum_absolute_float_gain_units": minimum_float_gain_units,
              "maximum_absolute_outcome_errors": metric_errors,
              "equal_cell_mean_consumer_surplus": rational_record(exact_mean_cs),
              "equal_cell_prize_probability": rational_record(exact_prize_probability),
              "equal_cell_expected_payment_euros": rational_record(exact_payment),
              "cells": cell_reports, "numpy_version": np.__version__,
              "elapsed_seconds": time.monotonic() - start,
              "production_model_sha256": hashlib.sha256((production / "experiment_model.py").read_bytes()).hexdigest(),
              "comparison_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "results/exact/comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
