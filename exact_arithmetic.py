#!/usr/bin/env python3
"""Finite-grid dynamic programme using integer and rational arithmetic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "experiment_config.json"


def as_units(x):
    y = Fraction(str(x)) * 100
    assert y.denominator == 1
    return y.numerator


def rational_record(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "decimal": float(value)}


def update_min(current, numerator, denominator, location):
    if numerator <= 0:
        return current
    if current is None or numerator * current[1] < current[0] * denominator:
        return (numerator, denominator, location)
    return current


def solve_integer_policy(upper, fee, price_count):
    """N_t(s)=L^(n-t) V_t(s); V is surplus net of future fees, in cents."""
    n = len(upper)
    top = max(0, max(upper))
    next_value = list(range(top + 1))
    powers = [price_count ** k for k in range(n + 1)]
    decisions = [None] * n
    gains = [None] * n
    thresholds = [None] * n
    value_at_zero = [None] * (n + 1)
    value_at_zero[n] = 0
    for t in range(n - 1, -1, -1):
        denominator = powers[n - t]
        high = upper[t]
        assert 0 < high < price_count  # True for every configured seller/type.
        cost_numerator = fee * denominator
        current = [0] * (top + 1)
        continue_here = [False] * (top + 1)
        gain = [0] * (top + 1)
        tail = 0
        for s in range(top, -1, -1):
            if s < high:
                tail += next_value[s + 1]
                transition_sum = (price_count - high + s) * next_value[s] + tail
            else:
                transition_sum = price_count * next_value[s]
            stop_numerator = s * denominator
            delta = transition_sum - cost_numerator - stop_numerator
            gain[s] = delta
            continue_here[s] = delta > 0  # Exact stop-on-zero convention.
            current[s] = stop_numerator + max(0, delta)
        assert all(gain[s + 1] <= gain[s] for s in range(top))
        threshold = next((s for s, flag in enumerate(continue_here) if not flag), top + 1)
        assert all(continue_here[s] == (s < threshold) for s in range(top + 1))
        decisions[t], gains[t], thresholds[t] = continue_here, gain, threshold
        value_at_zero[t] = current[0]
        next_value = current
    return decisions, gains, thresholds, value_at_zero, powers


def propagate_integer_paths(upper, distances, fee, gamma, outside, value,
                            price_count, decisions, gains, powers):
    """Unnormalized histories; early terminal paths are padded to n draws."""
    n, top = len(upper), len(decisions[0]) - 1
    mass, distance_mass = [0] * (top + 1), [0] * (top + 1)
    mass[0] = 1
    total_paths = powers[n]
    totals = {k: 0 for k in ("paths", "depth", "purchase", "surplus_units", "distance")}
    by_depth = [0] * (n + 1)
    reachable_states = [0] * n
    reachable_zero_gains = [0] * n
    min_reachable_positive = None
    min_reachable_negative = None
    for t in range(n + 1):
        padding = powers[n - t]
        active, active_dist = [0] * (top + 1), [0] * (top + 1)
        for s, count in enumerate(mass):
            if count == 0:
                assert distance_mass[s] == 0
                continue
            if t < n:
                reachable_states[t] += 1
                delta = gains[t][s]
                reachable_zero_gains[t] += int(delta == 0)
                location = {"stage": t, "state_units": s}
                min_reachable_positive = update_min(min_reachable_positive, delta, powers[n - t], location)
                min_reachable_negative = update_min(min_reachable_negative, -delta, powers[n - t], location)
            if t == n or not decisions[t][s]:
                padded = count * padding
                totals["paths"] += padded
                by_depth[t] += padded
                totals["depth"] += t * padded
                totals["purchase"] += int(s > 0) * padded
                totals["surplus_units"] += s * padded
                totals["distance"] += distance_mass[s] * padding
            else:
                active[s] = count
                active_dist[s] = distance_mass[s]
        if t == n:
            break
        high, distance = upper[t], distances[t]
        new_mass, new_dist = [0] * (top + 1), [0] * (top + 1)
        less_mass = 0
        for s in range(top + 1):
            retained_price_count = min(price_count, price_count - high + s)
            retained = retained_price_count * active[s]
            retained_dist = retained_price_count * active_dist[s]
            acquired = less_mass if 0 < s <= high else 0
            new_mass[s] = retained + acquired
            new_dist[s] = retained_dist + distance * acquired
            less_mass += active[s]
        assert sum(new_mass) == price_count * sum(active)
        mass, distance_mass = new_mass, new_dist
    assert totals["paths"] == total_paths
    assert sum(by_depth) == total_paths
    assert by_depth[0] == 0
    zero_surplus_paths = 1
    for high in upper:
        zero_surplus_paths *= price_count - high
    assert totals["purchase"] == total_paths - zero_surplus_paths
    totals["search_cost_units"] = fee * totals["depth"]
    totals["transport_cost_units"] = gamma * totals["distance"]
    totals["revenue_units"] = ((value - outside) * totals["purchase"]
                               - totals["surplus_units"] - totals["transport_cost_units"])
    totals["consumer_surplus_units"] = (outside * total_paths + totals["surplus_units"]
                                         - totals["search_cost_units"])
    totals["welfare_units"] = totals["consumer_surplus_units"] + totals["revenue_units"]
    return totals, by_depth, reachable_states, reachable_zero_gains, min_reachable_positive, min_reachable_negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=HERE / "results/exact")
    parser.add_argument("--quick", action="store_true", help="Three endpoint/midpoint types only; smoke test, not full audit.")
    args = parser.parse_args()
    started = time.monotonic()
    config_bytes = args.config.read_bytes()
    cfg = json.loads(config_bytes)
    assert cfg["price_step_points"] == cfg["outside_step_points"] == 0.01
    value = as_units(cfg["value_points"])
    pmin, pmax = as_units(cfg["price_min_points"]), as_units(cfg["price_max_points"])
    L = pmax - pmin + 1
    types = list(range(as_units(cfg["outside_min_points"]), as_units(cfg["outside_max_points"]) + 1))
    if args.quick:
        types = [types[0], types[len(types) // 2], types[-1]]
    args.out.mkdir(parents=True, exist_ok=True)
    threshold_rows, all_cells, details = [], [], []
    covered_states = feasible_states = optimal_reachable_states = 0
    zero_states = feasible_zeros = reachable_zeros = 0
    minimum_positive = minimum_negative = None
    minimum_feasible_positive = minimum_feasible_negative = None
    minimum_reachable_positive = minimum_reachable_negative = None
    for ranking, distances in cfg["rankings"].items():
        for cost_label, cost_points in cfg["inspection_costs_points"].items():
            fee = as_units(cost_points)
            for gamma_points in cfg["gammas"]:
                cell_started = time.monotonic()
                gamma = as_units(gamma_points)
                cell_total = None
                depth_totals = [0] * (len(distances) + 1)
                per_type_depth = []
                for outside in types:
                    upper = [value - outside - gamma * d - pmin for d in distances]
                    decisions, gains, thresholds, zero_values, powers = solve_integer_policy(upper, fee, L)
                    totals, by_depth, reach, reach_zero, reach_pos, reach_neg = propagate_integer_paths(
                        upper, distances, fee, gamma, outside, value, L, decisions, gains, powers)
                    assert zero_values[0] == totals["surplus_units"] - totals["search_cost_units"]
                    per_type_depth.append(totals["depth"])
                    if cell_total is None:
                        cell_total = {k: 0 for k in totals}
                    for k, v in totals.items():
                        cell_total[k] += v
                    depth_totals = [a + b for a, b in zip(depth_totals, by_depth)]
                    row = {"ranking": ranking, "cost_label": cost_label, "gamma": gamma_points,
                           "outside_units": outside}
                    row.update({f"finite_stop_units_{t}": v for t, v in enumerate(thresholds)})
                    threshold_rows.append(row)
                    metadata = {"ranking": ranking, "cost_label": cost_label,
                                "gamma": gamma_points, "outside_units": outside}
                    if reach_pos:
                        minimum_reachable_positive = update_min(minimum_reachable_positive, *reach_pos[:2], {**metadata, **reach_pos[2]})
                    if reach_neg:
                        minimum_reachable_negative = update_min(minimum_reachable_negative, *reach_neg[:2], {**metadata, **reach_neg[2]})
                    stage_details = []
                    for t, gain in enumerate(gains):
                        physically_feasible_top = 0 if t == 0 else max(upper[:t])
                        denominator = powers[len(distances) - t]
                        covered_states += len(gain)
                        feasible_states += physically_feasible_top + 1
                        optimal_reachable_states += reach[t]
                        all_zero = [s for s, delta in enumerate(gain) if delta == 0]
                        zero_states += len(all_zero)
                        feasible_zero = [s for s in all_zero if s <= physically_feasible_top]
                        feasible_zeros += len(feasible_zero)
                        reachable_zeros += reach_zero[t]
                        pos = min((x for x in gain if x > 0), default=None)
                        neg = min((-x for x in gain if x < 0), default=None)
                        if pos is not None:
                            loc = {**metadata, "stage": t, "state_units": gain.index(pos)}
                            minimum_positive = update_min(minimum_positive, pos, denominator, loc)
                        if neg is not None:
                            loc = {**metadata, "stage": t, "state_units": gain.index(-neg)}
                            minimum_negative = update_min(minimum_negative, neg, denominator, loc)
                        feasible_gain = gain[:physically_feasible_top + 1]
                        feasible_pos = min((x for x in feasible_gain if x > 0), default=None)
                        feasible_neg = min((-x for x in feasible_gain if x < 0), default=None)
                        if feasible_pos is not None:
                            minimum_feasible_positive = update_min(minimum_feasible_positive, feasible_pos, denominator, {**metadata, "stage": t, "state_units": gain.index(feasible_pos)})
                        if feasible_neg is not None:
                            minimum_feasible_negative = update_min(minimum_feasible_negative, feasible_neg, denominator, {**metadata, "stage": t, "state_units": gain.index(-feasible_neg)})
                        stop = thresholds[t]
                        stage_details.append({"stage": t, "all_states": len(gain),
                            "physically_feasible_states": physically_feasible_top + 1,
                            "optimal_policy_reachable_states": reach[t], "stop_state_units": stop,
                            "zero_gain_states": all_zero,
                            "gain_at_last_continue": rational_record(Fraction(gain[stop - 1], denominator)) if stop else None,
                            "gain_at_first_stop": rational_record(Fraction(gain[stop], denominator)) if stop < len(gain) else None})
                    details.append({**metadata, "stages": stage_details})
                assert len(set(per_type_depth)) == 1, "Expected depth varies across outside types"
                denominator = len(types) * L ** len(distances)
                expectation = {}
                for k, v in cell_total.items():
                    if k == "paths":
                        continue
                    out_key = k[:-6] if k.endswith("_units") else k
                    expectation[out_key] = rational_record(Fraction(v, denominator * (100 if k.endswith("_units") else 1)))
                expectation["conversion"] = expectation.pop("purchase")
                expectation["selected_distance_given_purchase"] = rational_record(Fraction(cell_total["distance"], cell_total["purchase"]))
                expectation["depth_distribution"] = [rational_record(Fraction(v, denominator)) for v in depth_totals]
                all_cells.append({"ranking": ranking, "cost_label": cost_label, "gamma": gamma_points,
                                  "expectations": expectation,
                                  "raw_sum_path_numerators": {k: str(v) for k, v in cell_total.items()},
                                  "base_denominator": str(denominator),
                                  "seconds": time.monotonic() - cell_started})
                print(f"{ranking} {cost_label} gamma={gamma_points}: {len(types)} types; "
                      f"CS={expectation['consumer_surplus']['decimal']:.14f}; "
                      f"{time.monotonic()-cell_started:.2f}s", flush=True)
    def encode_min(item):
        return None if item is None else {"gain_units": rational_record(Fraction(item[0], item[1])), "location": item[2]}
    summary = {"implementation_version": 1, "full_type_audit": not args.quick,
        "arithmetic": "Python arbitrary-precision integer; exact fractions for reporting",
        "price_count": L, "types_per_cell": len(types), "cells": len(all_cells),
        "decision_stages": len(distances), "total_bellman_state_comparisons": covered_states,
        "physically_feasible_state_comparisons": feasible_states,
        "optimal_policy_reachable_state_comparisons": optimal_reachable_states,
        "zero_continuation_gains_all_states": zero_states,
        "zero_continuation_gains_physically_feasible": feasible_zeros,
        "zero_continuation_gains_optimal_reachable": reachable_zeros,
        "minimum_positive_gain": encode_min(minimum_positive),
        "minimum_negative_gain_magnitude": encode_min(minimum_negative),
        "minimum_physically_feasible_positive_gain": encode_min(minimum_feasible_positive),
        "minimum_physically_feasible_negative_gain_magnitude": encode_min(minimum_feasible_negative),
        "minimum_optimal_reachable_positive_gain": encode_min(minimum_reachable_positive),
        "minimum_optimal_reachable_negative_gain_magnitude": encode_min(minimum_reachable_negative),
        "checks": ["Every Bellman gain compared to exact zero", "All stage gains weakly decrease in surplus",
                   "Threshold representation equals every computed decision", "Path mass conserved at every type/stage",
                   "Padded terminal path count equals L^4", "Conversion equals exact four-seller product",
                   "Consumer surplus equals independent backward Bellman value",
                   "Expected depth identical across all outside types within every cell"],
        "python": sys.version, "platform": platform.platform(),
        "config_path": str(args.config.resolve()), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "scope_note": "State aggregation covers the full finite probability law; no enumeration of all L^4 price profiles."}
    (args.out / "exact_cells.json").write_text(json.dumps(all_cells, indent=2) + "\n")
    (args.out / "state_certificates.json").write_text(json.dumps(details, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "exact_thresholds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
