"""Artificial validation only. No empirical participant data are used or saved."""

import copy
import csv
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import t as student_t

import experiment_analysis as an


def fixture_rows(pairs=24, rounds=3):
    """Deterministic, manually specified cells; this is NOT an iid trial sample."""
    rows = []
    for b in range(pairs):
        q = (b//6) % 2
        for r in range(1, rounds+1):
            c, g = divmod((b+r-1) % 6, 3)
            for a in range(2):
                base = 1+c
                depth = ([base+q, base, base-1+q] if a == 0
                         else [base, base+q, base+1+q])[g]
                rows.append({
                    "pair_id": f"pair{b:03d}", "participant_id": f"person{b:03d}_{a}",
                    "rank": a, "round": r, "cost": c, "gamma": an.GAMMAS[g],
                    "T": depth, "depth_observed": True, "terminal_complete": True, "observed_opens": depth,
                    "purchase": (q+a) % 2, "induced_CS": 10-a*(2+g)+c+r/10+q,
                })
    return rows


def fixture_accounting_rows(pairs=24, rounds=3):
    """Artificial but internally reconciled actual-choice accounting records."""
    rows = fixture_rows(pairs, rounds)
    for row in rows:
        sc = an.FEE_LEVELS[row["cost"]]*row["T"]
        tc = row["gamma"] if row["purchase"] else 0
        rev = 50 if row["purchase"] else 0
        cs = 120-rev-tc-sc if row["purchase"] else 50-sc
        row.update(entry=int(row["T"] > 0), SC=sc, TC=tc, REV=rev,
                   induced_CS=cs, W=cs+rev)
    return rows


def explicit_full_matrix(data, outcome, weights, config):
    """Independent full X/H/cluster-eigendecomposition definition of CR2."""
    keep = np.isfinite(data[outcome])
    y = data[outcome][keep]
    a, c, g, r = [data[key][keep] for key in ("rank", "cost", "gamma", "round")]
    b = data["pair"][keep]
    cells = []
    for aa in range(2):
        for cc in range(2):
            for gg in range(3):
                for rr in range(config.rounds):
                    cells.append((a == aa) & (c == cc) & (g == gg) & (r == rr))
    X = np.asarray(cells, dtype=float).T
    bread = np.linalg.inv(np.einsum("ni,nj->ij", X, X))
    H = np.einsum("ni,ij,mj->nm", X, bread, X, optimize=False)
    residual_maker = np.eye(len(y))-H
    estimate = float(np.einsum("i,ij,nj,n->", weights, bread, X, y, optimize=False))
    K = np.zeros((config.pairs, len(y)))
    for pair in range(config.pairs):
        index = np.flatnonzero(b == pair)
        if not len(index):
            continue
        matrix = np.eye(len(index))-H[np.ix_(index, index)]
        eigenvalues, vectors = np.linalg.eigh(matrix)
        A = np.einsum("ij,kj->ik", vectors/np.sqrt(eigenvalues), vectors)
        left = np.einsum("i,ij,nj->n", weights, bread, X[index], optimize=False)
        K[pair] = np.einsum("i,ij,jn->n", left, A, residual_maker[index], optimize=False)
    scores = np.einsum("bn,n->b", K, y)
    G = np.einsum("bn,cn->bc", K, K)
    return {"estimate": estimate, "variance": float(np.dot(scores, scores)),
            "df": float(np.trace(G)**2/np.einsum("bc,bc->", G, G)), "G": G}


def manual_raw_interaction(data, rounds):
    result = 0.0
    for r in range(rounds):
        for c in range(2):
            for a in range(2):
                for g, sign in [(0, -1), (2, 1)]:
                    chosen = ((data["rank"] == a) & (data["cost"] == c)
                              & (data["gamma"] == g) & (data["round"] == r)
                              & np.isfinite(data["T"]))
                    result += (2*a-1)*sign*data["T"][chosen].mean()/(2*rounds)
    return result


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = an.Config(rounds=2, pairs=12, ri_draws=99)
        self.rows = fixture_rows(12, 2)

    def test_zero_depth_and_incomplete_pair_retained(self):
        zeros = sum(row["T"] == 0 for row in self.rows)
        self.assertGreater(zeros, 0)
        for row in self.rows:
            if row["participant_id"] == "person000_1":
                row.update(T="", depth_observed=False, terminal_complete=False, observed_opens="", purchase="", induced_CS="")
        data = an.validate_rows(self.rows, self.cfg)
        self.assertEqual(len(data["T"]), 48)
        self.assertEqual(np.sum(data["T"] == 0), zeros)
        self.assertEqual(np.sum(~np.isfinite(data["T"])), 2)
        self.assertEqual(len(np.unique(data["pair"])), 12)

    def test_omitted_slot_and_duplicate_rejected(self):
        with self.assertRaises(an.ValidationError):
            an.validate_rows(self.rows[:-1], self.cfg)
        with self.assertRaises(an.ValidationError):
            an.validate_rows(self.rows+[self.rows[0]], self.cfg)

    def test_assignment_inconsistency_rejected(self):
        for field, value in [("gamma", 4), ("cost", 1), ("rank", 1)]:
            bad = copy.deepcopy(self.rows)
            bad[0][field] = value
            with self.assertRaises(an.ValidationError):
                an.validate_rows(bad, self.cfg)

    def test_partial_depth_and_impossible_lower_bound_rejected(self):
        bad = copy.deepcopy(self.rows)
        bad[0]["depth_observed"] = False
        with self.assertRaises(an.ValidationError):
            an.validate_rows(bad, self.cfg)
        bad = copy.deepcopy(self.rows)
        bad[0]["observed_opens"] = 4
        with self.assertRaises(an.ValidationError):
            an.validate_rows(bad, self.cfg)

    def test_invalid_values_rejected(self):
        for field, value in [("T", 1.5), ("purchase", 2), ("induced_CS", "inf"),
                             ("round", 0), ("gamma", 1), ("terminal_complete", "maybe")]:
            bad = copy.deepcopy(self.rows)
            bad[0][field] = value
            with self.assertRaises(an.ValidationError):
                an.validate_rows(bad, self.cfg)

    def test_administrative_partial_depth_is_not_exact_completed_task_T(self):
        for k in range(4):
            rows = copy.deepcopy(self.rows)
            rows[0].update(T=k, depth_observed=True, terminal_complete=False,
                           observed_opens=k, actual_final_T=k, purchase="", induced_CS="")
            with self.assertRaisesRegex(an.ValidationError, "completed-task T"):
                an.validate_rows(rows, self.cfg)

    def test_verified_cap_four_retained_without_economic_choice(self):
        self.rows[0].update(T=4, observed_opens=4, terminal_complete=False,
                            actual_final_T=4, purchase="", induced_CS="")
        data = an.validate_rows(self.rows, self.cfg)
        self.assertEqual(np.isfinite(data["T"]).sum(), 48)
        self.assertEqual(np.sum(data["depth_observed"] & ~data["terminal_complete"]), 1)
        self.assertEqual(data["T"][0], 4)
        for lower_bound in ("", 0, 3):
            rows = copy.deepcopy(self.rows)
            rows[0]["observed_opens"] = lower_bound
            with self.assertRaisesRegex(an.ValidationError, "verified observed_opens=4"):
                an.validate_rows(rows, self.cfg)

    def test_actual_final_depth_is_descriptive_and_does_not_impute_target(self):
        self.rows[0].update(T="", depth_observed=False, terminal_complete=False,
                            observed_opens=1, actual_final_T=2, purchase="", induced_CS="")
        data = an.validate_rows(self.rows, self.cfg)
        self.assertTrue(np.isnan(data["T"][0]))
        self.assertEqual(data["actual_final_T"][0], 2)
        self.assertEqual(data["observed_opens"][0], 1)
        report = an.analyse(data, self.cfg)
        self.assertEqual(report["analysis_version"], "reference-3.0-primary-joint-signs")
        self.assertEqual(report["accounting"]["observed_T"], 47)
        self.assertEqual(report["actual_final_depth_descriptive"]["observed_records"], 1)
        self.assertEqual(report["actual_final_depth_descriptive"]["observed_mean"], 2)
        self.assertFalse(report["actual_final_depth_descriptive"]["used_to_impute_target_outcomes"])
        self.assertEqual(report["HT_depth_bounds"]["interaction"]["missing_T_slots"], 1)
        json.dumps(report, allow_nan=False)

    def test_actual_final_depth_support_and_log_consistency(self):
        for actual, opens in ((1.5, 0), (-1, 0), (5, 0), (1, 2)):
            rows = copy.deepcopy(self.rows)
            rows[0].update(T="", depth_observed=False, terminal_complete=False,
                           actual_final_T=actual, observed_opens=opens, purchase="", induced_CS="")
            with self.assertRaisesRegex(an.ValidationError, "actual_final_T"):
                an.validate_rows(rows, self.cfg)
        rows = copy.deepcopy(self.rows)
        rows[0]["actual_final_T"] = 4
        with self.assertRaisesRegex(an.ValidationError, "disagree"):
            an.validate_rows(rows, self.cfg)

    def test_choice_outcomes_require_economic_terminal_choice(self):
        for outcome in ("purchase", "induced_CS"):
            rows = copy.deepcopy(self.rows)
            rows[0].update(T=4, observed_opens=4, terminal_complete=False, purchase="", induced_CS="")
            rows[0][outcome] = 0
            with self.assertRaises(an.ValidationError):
                an.validate_rows(rows, self.cfg)

    def test_CS_support_inclusive_and_rejects_outside_values(self):
        for value in (-32, 99):
            rows = copy.deepcopy(self.rows)
            rows[0]["induced_CS"] = value
            an.validate_rows(rows, self.cfg)
        for value in (-32.0001, 99.0001):
            rows = copy.deepcopy(self.rows)
            rows[0]["induced_CS"] = value
            with self.assertRaises(an.ValidationError):
                an.validate_rows(rows, self.cfg)

    def test_CSV_roundtrip_with_zero_and_missing(self):
        self.rows[0].update(T="", depth_observed=False, terminal_complete=False,
                            observed_opens=0, purchase="", induced_CS="")
        with tempfile.TemporaryDirectory(prefix="artificial_analysis_unit_test_") as directory:
            path = Path(directory)/"artificial_validation_only.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=an.REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerows(self.rows)
            data = an.load_csv(path, self.cfg)
        self.assertEqual(len(data["T"]), 48)
        self.assertEqual(np.sum(~np.isfinite(data["T"])), 1)


class EstimationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = an.Config(rounds=3, pairs=24, ri_draws=99)
        self.rows = fixture_rows(24, 3)
        self.data = an.validate_rows(self.rows, self.cfg)

    def test_manually_known_means(self):
        known = [("T", "interaction", 2, 2.5), ("T", "near_simple", 2, -1.0),
                 ("T", "far_simple", 2, 1.5), ("T", "rank_at_gamma", 0, -0.5),
                 ("T", "cost_pooled", 2, 1.0), ("T", "friction_pooled", 2, 0.25),
                 ("induced_CS", "near_CS_positive_gamma", 2, 3.5)]
        for outcome, name, gamma, expected in known:
            result = an.cr2(self.data, outcome, an.contrast(3, name, gamma), self.cfg)
            self.assertAlmostEqual(result["estimate"], expected, places=12)

    def test_full_matrix_with_unequal_missing_denominators(self):
        for index in (1, 14, 61, 82):
            # Depth logging can be missing despite a documented economic choice.
            self.rows[index].update(T="", depth_observed=False, observed_opens=0)
        data = an.validate_rows(self.rows, self.cfg)
        for name, gamma in [("interaction", 2), ("near_simple", 2), ("far_simple", 2),
                            ("rank_at_gamma", 0), ("cost_pooled", 2), ("three_way", 2)]:
            weights = an.contrast(3, name, gamma)
            fast = an.cr2(data, "T", weights, self.cfg, geometry=True)
            full = explicit_full_matrix(data, "T", weights, self.cfg)
            for key in ("estimate", "variance", "df"):
                self.assertAlmostEqual(fast[key], full[key], places=10)
            np.testing.assert_allclose(fast["G"], full["G"], atol=1e-12)
            self.assertLessEqual(fast["df"], self.cfg.pairs-1+1e-10)

    def test_empty_and_singleton_are_not_silently_filled(self):
        for keep_count in (0, 1):
            data = copy.deepcopy(self.data)
            selected = np.flatnonzero((data["rank"] == 0) & (data["cost"] == 0)
                                      & (data["gamma"] == 0) & (data["round"] == 0))
            data["T"][selected[keep_count:]] = np.nan
            result = an.safe_cr2(data, "T", an.contrast(3, "interaction"), self.cfg)
            self.assertEqual(result["status"], "not_estimable")
        data = copy.deepcopy(self.data)
        data["T"][:] = np.nan
        self.assertEqual(an.safe_cr2(data, "T", an.contrast(3, "interaction"), self.cfg)["status"],
                         "not_estimable")
        self.assertEqual(an.sharp_path_ri(data, self.cfg)["status"], "not_estimable")

    def test_zero_variance_is_not_an_automatic_discovery(self):
        data = copy.deepcopy(self.data)
        data["T"][:] = 2
        result = an.cr2(data, "T", an.contrast(3, "interaction"), self.cfg)
        self.assertEqual(result["status"], "zero_variance_no_automatic_test")
        self.assertIsNone(result["p_greater"])

    def test_identical_paired_binary_conversion_is_numerically_degenerate(self):
        cfg = an.Config(rounds=24, pairs=150, ri_draws=99)
        data = an.validate_rows(fixture_rows(150, 24), cfg)
        rng = np.random.default_rng(129)
        pair_conversion = rng.binomial(1, .72, size=(150, 24))
        data["purchase"] = pair_conversion[data["pair"], data["round"]].astype(float)
        for gamma in range(3):
            result = an.cr2(data, "purchase", an.contrast(24, "rank_at_gamma", gamma), cfg)
            self.assertLessEqual(abs(result["estimate"]), 1e-12)
            self.assertLessEqual(result["se"], result["numerical_se_tolerance"])
            self.assertEqual(result["status"], "zero_variance_no_automatic_test")
            self.assertIsNone(result["p_greater"])
            self.assertIsNone(result["ci95"])
            self.assertIsNone(an.tost(result, .05)["p"])

    def test_genuine_small_nonzero_variance_above_tolerance_is_retained(self):
        data = copy.deepcopy(self.data)
        data["induced_CS"] = 1e-10*data["T"]
        weights = an.contrast(3, "interaction")
        reference = an.cr2(self.data, "T", weights, self.cfg)
        result = an.cr2(data, "induced_CS", weights, self.cfg)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["se"], result["numerical_se_tolerance"])
        self.assertAlmostEqual(result["se"]/reference["se"], 1e-10, delta=1e-22)
        self.assertAlmostEqual(result["t"], reference["t"], places=10)

    def test_nonzero_below_fixed_numerical_resolution_is_not_tested(self):
        data = copy.deepcopy(self.data)
        data["induced_CS"] = 1e-13*data["T"]
        result = an.cr2(data, "induced_CS", an.contrast(3, "interaction"), self.cfg)
        self.assertGreater(result["se"], 0)
        self.assertEqual(result["status"], "zero_variance_no_automatic_test")

    def test_tost_and_holm_manually(self):
        result = {"status": "ok", "estimate": 0.1, "se": 0.04, "df": 20}
        tost = an.tost(result, 0.25)
        expected = max(student_t.sf(8.75, 20), student_t.cdf(-3.75, 20))
        self.assertAlmostEqual(tost["p"], expected, places=14)
        adjusted = an.holm(dict(a=0.001, b=0.02, c=0.03, d=0.04))
        np.testing.assert_allclose([adjusted[x]["holm_p"] for x in "abcd"], [.004, .06, .06, .06])
        unavailable = an.holm(dict(a=0.001, b=None, c=0.03, d=0.04))
        self.assertEqual(unavailable["a"]["holm_p"], .004)
        self.assertEqual(unavailable["b"]["status"], "not_estimable")

    def test_tost_rechecks_numerical_tolerance_if_called_with_stale_ok_flag(self):
        for se in (0, 1e-18, 1e-12):
            result = {"status": "ok", "estimate": 0, "se": se, "df": 149}
            self.assertIsNone(an.tost(result, .05)["p"])

    def test_full_report_family_and_outcome_specific_missingness(self):
        data = copy.deepcopy(self.data)
        data["purchase"][:] = np.nan
        report = an.analyse(data, self.cfg)
        self.assertEqual(report["interaction_supporting"]["registered_alternative"], "theta > 0")
        self.assertEqual(len(report["secondary_Holm_family"]), 4)
        self.assertIsNone(report["conversion_joint_TOST"]["p"])
        self.assertEqual(report["interaction_supporting"]["status"], "ok")
        self.assertAlmostEqual(report["supplementary_sharp_path_RI"]["observed_statistic"],
                               report["interaction_supporting"]["estimate"], places=12)

    def test_all_outcomes_missing_produces_accounting_and_bounds_not_failure(self):
        data = copy.deepcopy(self.data)
        for outcome in ("T", "purchase", "induced_CS"):
            data[outcome][:] = np.nan
        data["observed_opens"][:] = 0
        report = an.analyse(data, self.cfg)
        self.assertEqual(report["interaction_supporting"]["status"], "not_estimable")
        self.assertEqual(report["accounting"]["assigned_participants"], 48)
        self.assertEqual(report["HT_depth_bounds"]["interaction"]["missing_T_slots"], 144)
        json.dumps(report, allow_nan=False)

    def test_candidate_size_and_18_round_configuration(self):
        for rounds in (18, 24):
            cfg = an.Config(rounds=rounds, pairs=150, ri_draws=99)
            data = an.validate_rows(fixture_rows(150, rounds), cfg)
            report = an.analyse(data, cfg)
            self.assertEqual(report["accounting"]["assigned_slots"], 300*rounds)
            self.assertEqual(report["interaction_supporting"]["status"], "ok")
            self.assertEqual(len(report["cell_means"]["T"]), 12*rounds+12)
            json.dumps(report, allow_nan=False)


class BoundsAndRITests(unittest.TestCase):
    def setUp(self):
        self.cfg = an.Config(rounds=2, pairs=12, ri_draws=999, ri_seed=42,
                             gamma_probabilities=(1/3, 1/3, 1/3))
        self.data = an.validate_rows(fixture_rows(12, 2), self.cfg)

    def test_HT_extrema_by_exhaustive_missing_value_completions(self):
        data = copy.deepcopy(self.data)
        index = np.flatnonzero((data["gamma"] == 2) & (data["round"] == 0))[:2]
        data["T"][index] = np.nan
        data["observed_opens"][index] = [1, 2]
        bound = an.ht_bounds(data, self.cfg)
        sign = (data["gamma"] == 2).astype(float)-(data["gamma"] == 0)
        weight = 3/self.cfg.rounds*(2*data["rank"]-1)*sign/self.cfg.pairs
        completions = []
        for first, second in itertools.product(range(1, 5), range(2, 5)):
            y = data["T"].copy()
            y[index] = [first, second]
            completions.append(float(np.dot(weight, y)))
        np.testing.assert_allclose(bound["exact_HT_completion_extrema"],
                                   [min(completions), max(completions)], atol=1e-12)
        self.assertEqual(bound["missing_T_slots"], 2)

    def test_no_missing_bounds_equal_exact_HT_not_Hajek(self):
        bound = an.ht_bounds(self.data, self.cfg)
        self.assertAlmostEqual(bound["exact_HT_completion_extrema"][0],
                               bound["exact_HT_completion_extrema"][1], places=12)
        self.assertGreater(bound["finite_Hoeffding_envelope"][1],
                           bound["exact_HT_completion_extrema"][1])

    def test_vectorised_RI_matches_manual_rank_flips_with_missingness(self):
        cfg = an.Config(rounds=2, pairs=24, ri_draws=99)
        data = an.validate_rows(fixture_rows(24, 2), cfg)
        data["T"][[1, 17, 43]] = np.nan
        components = an._ri_components(data, cfg)
        rng = np.random.default_rng(812)
        signs = rng.integers(0, 2, size=(30, cfg.pairs))*2-1
        fast = an._ri_statistics(signs, components)
        for index, row in enumerate(signs):
            altered = copy.deepcopy(data)
            changed = row[data["pair"]] == -1
            altered["rank"][changed] = 1-altered["rank"][changed]
            self.assertAlmostEqual(fast[index], manual_raw_interaction(altered, cfg.rounds), places=12)

    def test_all_small_assignments_and_reproducibility(self):
        cfg = an.Config(rounds=1, pairs=6, ri_draws=999, ri_seed=91)
        data = an.validate_rows(fixture_rows(6, 1), cfg)
        signs = np.asarray(list(itertools.product((-1, 1), repeat=6)))
        fast = an._ri_statistics(signs, an._ri_components(data, cfg))
        for i, sign in enumerate(signs):
            altered = copy.deepcopy(data)
            change = sign[data["pair"]] == -1
            altered["rank"][change] = 1-altered["rank"][change]
            self.assertAlmostEqual(fast[i], manual_raw_interaction(altered, 1), places=12)
        one = an.sharp_path_ri(data, cfg)
        two = an.sharp_path_ri(data, cfg)
        self.assertEqual(one, two)
        self.assertGreaterEqual(one["p_plus_one"], 1/1000)
        self.assertLessEqual(one["p_plus_one"], 1)

    def test_RI_refuses_undefined_counterfactual_cells(self):
        data = copy.deepcopy(self.data)
        chosen = (data["cost"] == 0) & (data["gamma"] == 0) & (data["round"] == 0)
        for pair in np.unique(data["pair"][chosen]):
            data["T"][chosen & (data["pair"] == pair) & (data["rank"] == pair % 2)] = np.nan
        result = an.sharp_path_ri(data, self.cfg)
        self.assertEqual(result["status"], "not_estimable")
        self.assertIsNone(result["p_plus_one"])


class ExploratoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = an.Config(rounds=4, pairs=24, ri_draws=99)

    def test_derived_entry_uses_verified_prefix_while_SC_requires_target_T(self):
        rows = fixture_accounting_rows(24, 4)
        rows[0].update(T="", depth_observed=False, actual_final_T=2, observed_opens=1,
                       terminal_complete=False, purchase="", induced_CS="", TC="", REV="", W="", entry=1, SC="")
        rows[1].update(T="", depth_observed=False, terminal_complete=False,
                       purchase="", induced_CS="", observed_opens=0, actual_final_T=0,
                       entry="", SC="", TC="", REV="", W="")
        rows[2].update(T=4, depth_observed=True, observed_opens=4, actual_final_T=4,
                       terminal_complete=False, purchase="", induced_CS="", TC="", REV="", W="",
                       entry=1, SC=4*an.FEE_LEVELS[rows[2]["cost"]])
        data = an.derive_depth_outcomes(an.validate_rows(rows, self.cfg))
        known = np.isfinite(data["T"])
        np.testing.assert_array_equal(data["entry"][known], data["T"][known] > 0)
        np.testing.assert_allclose(data["SC"][known],
                                   np.asarray(an.FEE_LEVELS)[data["cost"]][known]*data["T"][known])
        self.assertEqual(np.sum(~np.isfinite(data["SC"])), 2)
        self.assertEqual(np.sum(np.isfinite(data["SC"]) & ~data["terminal_complete"]), 1)
        positive_prefix = ~known & (data["observed_opens"] > 0)
        self.assertEqual(data["entry"][positive_prefix].tolist(), [1])
        zero_prefix = ~known & (data["observed_opens"] == 0)
        self.assertTrue(np.isnan(data["entry"][zero_prefix]).all())

    def test_actual_depth_alone_cannot_supply_entry_or_target_SC(self):
        for field, value, opens in (("entry", 0, 0), ("entry", 1, 0), ("SC", 1, 1)):
            rows = fixture_accounting_rows(24, 4)
            rows[0].update(T="", depth_observed=False, actual_final_T=2, observed_opens=opens,
                           terminal_complete=False, purchase="", induced_CS="", TC="", REV="", W="", entry="", SC="")
            rows[0][field] = value
            with self.assertRaises(an.ValidationError):
                an.validate_rows(rows, self.cfg)

    def test_optional_identities_and_support_validation(self):
        rows = fixture_accounting_rows(24, 4)
        an.validate_rows(rows, self.cfg)
        for field, delta in (("entry", 1), ("SC", 1), ("TC", 1), ("REV", 1), ("W", 1)):
            bad = copy.deepcopy(rows)
            bad[0][field] += delta
            with self.assertRaises(an.ValidationError):
                an.validate_rows(bad, self.cfg)
        for field, value in (("TC", -1), ("TC", 16.01), ("REV", 120.01), ("W", 32.99), ("W", 119.01)):
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(an.ValidationError):
                an.validate_rows(bad, self.cfg)

    def test_optional_admin_choice_values_are_rejected_even_at_verified_cap(self):
        for field in ("TC", "REV", "W"):
            rows = fixture_accounting_rows(24, 4)
            rows[0].update(T=4, observed_opens=4, terminal_complete=False,
                           purchase="", induced_CS="", TC="", REV="", W="", entry=1,
                           SC=4*an.FEE_LEVELS[rows[0]["cost"]])
            rows[0][field] = 50 if field == "W" else 0
            with self.assertRaises(an.ValidationError):
                an.validate_rows(rows, self.cfg)

    def test_fee_increment_and_period_vectors_have_known_values(self):
        data = an.validate_rows(fixture_rows(24, 4), self.cfg)
        q = (data["pair"]//6) % 2
        c, g, a, period = (data[key] for key in ("cost", "gamma", "rank", "round"))
        near = np.where(g == 0, 2, np.where(g == 2, np.where(period < 2, 1, 2), 1))
        far = np.where(g == 2, 2, 1)
        data["T"] = (np.where(a == 0, near, far)+q+c).astype(float)
        report = an.analyse(data, self.cfg)
        extra = report["depth_exploratory"]
        for fee in ("1", "4"):
            self.assertAlmostEqual(extra["endpoint_effects_by_fee"][fee]["near_simple"]["estimate"], -.5)
            self.assertAlmostEqual(extra["endpoint_effects_by_fee"][fee]["far_simple"]["estimate"], 1)
            self.assertAlmostEqual(extra["endpoint_effects_by_fee"][fee]["interaction"]["estimate"], 1.5)
        self.assertAlmostEqual(extra["friction_increments"]["0_to_2"]["near_simple"]["estimate"], -1)
        self.assertAlmostEqual(extra["friction_increments"]["2_to_4"]["near_simple"]["estimate"], .5)
        self.assertAlmostEqual(extra["friction_increments"]["0_to_2"]["interaction"]["estimate"], 1)
        self.assertAlmostEqual(extra["friction_increments"]["2_to_4"]["interaction"]["estimate"], .5)
        halves = extra["period_halves"]
        self.assertAlmostEqual(halves["first_half"]["estimate"], 2)
        self.assertAlmostEqual(halves["last_half"]["estimate"], 1)
        self.assertAlmostEqual(halves["last_minus_first"]["estimate"], -1)
        self.assertAlmostEqual(report["interaction_supporting"]["estimate"], 1.5)
        weights = an.contrast(4, "interaction")
        np.testing.assert_allclose(weights,
            .5*(an.restrict_periods(weights, 4, 0, 2)+an.restrict_periods(weights, 4, 2, 4)))
        np.testing.assert_allclose(an.restrict_cost(weights, 4, 1)-an.restrict_cost(weights, 4, 0),
                                   an.contrast(4, "three_way"))

    def test_optional_accounting_full_cell_summaries_and_fixed_confirmatory_family(self):
        data = an.validate_rows(fixture_accounting_rows(24, 4), self.cfg)
        report = an.analyse(data, self.cfg)
        for outcome in ("entry", "SC", "TC", "REV", "W"):
            self.assertIn(outcome, report["exploratory_outcomes"])
            self.assertEqual(len(report["cell_means"][outcome]), 60)
        self.assertEqual(len(report["secondary_Holm_family"]), 4)
        self.assertEqual(set(report["depth_exploratory"]["ranking_difference_at_gamma"]), {"0", "2", "4"})
        json.dumps(report, allow_nan=False)

    def test_rendered_sensitivity_never_infers_missing_acknowledgements_from_T(self):
        rows = fixture_rows(24, 4)
        for row in rows:
            row.update(rendered_T="", rendered_depth_observed=False)
        report = an.analyse(an.validate_rows(rows, self.cfg), self.cfg)
        self.assertEqual(report["rendered_revelation_sensitivity"]["status"], "not_available")
        self.assertEqual(report["accounting"]["certified_rendered_depth_records"], 0)
        self.assertGreater(report["accounting"]["observed_T"], 0)

    def test_certified_rendered_sensitivity_matches_known_outcomes_and_keeps_family(self):
        rows = fixture_rows(24, 4)
        for row in rows:
            row.update(rendered_T=row["T"], rendered_depth_observed=True)
        report = an.analyse(an.validate_rows(rows, self.cfg), self.cfg)
        self.assertAlmostEqual(report["rendered_revelation_sensitivity"]["estimate"],
                               report["interaction_supporting"]["estimate"], places=12)
        self.assertEqual(len(report["secondary_Holm_family"]), 4)
        self.assertEqual(report["accounting"]["certified_rendered_depth_records"], 192)

    def test_rendered_joint_signs_and_both_components_match_identical_committed_target(self):
        rows = fixture_rows(24, 4)
        committed = an.analyse(an.validate_rows(rows, self.cfg), self.cfg)
        for row in rows:
            row.update(rendered_T=row["T"], rendered_depth_observed=True)
        report = an.analyse(an.validate_rows(rows, self.cfg), self.cfg)
        sensitivity = report["rendered_revelation_sensitivity"]
        for name in ("near_simple", "far_simple", "interaction_supporting"):
            for field in ("estimate", "se", "df", "p_less", "p_greater"):
                self.assertAlmostEqual(sensitivity[name][field], committed[name][field], places=12)
        for field in ("primary_p", "reject_primary", "component_pvalues", "unavailable_components"):
            self.assertEqual(sensitivity["primary_joint_signs"][field], committed["primary_joint_signs"][field])
        self.assertFalse(sensitivity["may_replace_committed_primary"])
        self.assertFalse(sensitivity["primary_joint_signs"]["may_replace_committed_primary"])
        self.assertEqual(report["primary_joint_signs"], committed["primary_joint_signs"])
        self.assertEqual(report["secondary_Holm_family"], committed["secondary_Holm_family"])

    def test_rendered_one_unavailable_component_prevents_sensitivity_success(self):
        rows = fixture_rows(24, 4)
        committed = an.analyse(an.validate_rows(rows, self.cfg), self.cfg)
        for row in rows:
            row.update(rendered_T=row["T"] if row["rank"] == 1 else "",
                       rendered_depth_observed=row["rank"] == 1)
        report = an.analyse(an.validate_rows(rows, self.cfg), self.cfg)
        sensitivity = report["rendered_revelation_sensitivity"]
        self.assertEqual(sensitivity["near_simple"]["status"], "not_estimable")
        self.assertEqual(sensitivity["far_simple"]["status"], "ok")
        self.assertEqual(sensitivity["primary_joint_signs"]["primary_p"], 1.)
        self.assertFalse(sensitivity["primary_joint_signs"]["reject_primary"])
        self.assertEqual(sensitivity["primary_joint_signs"]["unavailable_components"], ["near_negative"])
        self.assertEqual(report["primary_joint_signs"], committed["primary_joint_signs"])

    def test_rendered_no_data_reports_unavailable_components_without_T_imputation(self):
        for include_empty_columns in (False, True):
            rows = fixture_rows(24, 4)
            if include_empty_columns:
                for row in rows:
                    row.update(rendered_T="", rendered_depth_observed=False)
            report = an.analyse(an.validate_rows(rows, self.cfg), self.cfg)
            sensitivity = report["rendered_revelation_sensitivity"]
            self.assertEqual(sensitivity["status"], "not_available")
            for name in ("near_simple", "far_simple", "interaction_supporting"):
                self.assertEqual(sensitivity[name]["status"], "not_available")
                self.assertNotIn("estimate", sensitivity[name])
            self.assertEqual(sensitivity["primary_joint_signs"]["primary_p"], 1.)
            self.assertFalse(sensitivity["primary_joint_signs"]["reject_primary"])
            self.assertEqual(len(sensitivity["primary_joint_signs"]["unavailable_components"]), 2)
            json.dumps(report, allow_nan=False)

    def test_rendered_count_requires_certificate_and_cannot_exceed_exact_T(self):
        for count, flag in ((1, False), ("", True), (1.5, True), (4, True)):
            rows = fixture_rows(24, 4)
            rows[0].update(rendered_T=count, rendered_depth_observed=flag)
            with self.assertRaises(an.ValidationError):
                an.validate_rows(rows, self.cfg)

    def test_administrative_rendered_prefix_is_not_completed_task_outcome(self):
        for count in range(4):
            rows = fixture_rows(24, 4)
            rows[0].update(T="", depth_observed=False, terminal_complete=False,
                           observed_opens=count, actual_final_T=count, purchase="", induced_CS="",
                           rendered_T=count, rendered_depth_observed=True)
            with self.assertRaisesRegex(an.ValidationError, "completed-task rendered_T"):
                an.validate_rows(rows, self.cfg)

    def test_four_certified_rendered_revelations_fix_target_without_choice(self):
        rows = fixture_rows(24, 4)
        rows[0].update(T=4, depth_observed=True, observed_opens=4, actual_final_T=4,
                       terminal_complete=False, purchase="", induced_CS="",
                       rendered_T=4, rendered_depth_observed=True)
        data = an.validate_rows(rows, self.cfg)
        self.assertEqual(data["rendered_T"][0], 4)
        self.assertFalse(data["terminal_complete"][0])
        # A four-open committed cap does not certify a partial rendered outcome.
        rows[0]["rendered_T"] = 3
        with self.assertRaisesRegex(an.ValidationError, "completed-task rendered_T"):
            an.validate_rows(rows, self.cfg)

    def test_rendered_count_cannot_exceed_descriptive_actual_final_count(self):
        rows = fixture_rows(24, 4)
        rows[0].update(T="", depth_observed=False, observed_opens=1, actual_final_T=1,
                       rendered_T=2, rendered_depth_observed=True)
        with self.assertRaisesRegex(an.ValidationError, "cannot exceed actual_final_T"):
            an.validate_rows(rows, self.cfg)


class JointPrimaryAndAllocationTests(unittest.TestCase):
    @staticmethod
    def component(estimate, se=.2, df=30):
        return {"status": "ok", "estimate": estimate, "se": se, "df": df,
                "p_less": float(student_t.cdf(estimate/se, df)),
                "p_greater": float(student_t.sf(estimate/se, df))}

    def test_primary_requires_both_signs_even_with_positive_interaction(self):
        for near, far in ((1, 2), (-2, -1), (0, 1), (-1, 0)):
            with self.subTest(near=near, far=far):
                self.assertGreater(far-near, 0)
                result = an.joint_signs_primary(self.component(near), self.component(far))
                self.assertFalse(result["reject_primary"])
                self.assertGreaterEqual(result["primary_p"], .5)

    def test_opposing_signs_pass_only_when_both_components_pass(self):
        near, far = self.component(-1), self.component(.8)
        result = an.joint_signs_primary(near, far)
        self.assertEqual(result["primary_p"], max(near["p_less"], far["p_greater"]))
        self.assertTrue(result["reject_primary"])
        self.assertFalse(result["decision_uses_Holm_family"])
        weak = an.joint_signs_primary(self.component(-.05), far)
        self.assertFalse(weak["reject_primary"])
        self.assertEqual(weak["primary_p"], self.component(-.05)["p_less"])

    def test_unavailable_component_has_effective_one_and_no_success(self):
        unavailable = ({"status": "not_estimable"},
                       {"status": "zero_variance_no_automatic_test", "p_less": 0., "p_greater": 0.},
                       {"status": "ok", "p_less": None, "p_greater": None},
                       {"status": "ok", "p_less": float("nan"), "p_greater": float("nan")})
        for bad in unavailable:
            for near, far in ((bad, self.component(1)), (self.component(-1), bad)):
                with self.subTest(bad=bad, near=near):
                    result = an.joint_signs_primary(near, far)
                    self.assertEqual(result["primary_p"], 1.)
                    self.assertEqual(result["status"], "not_estimable")
                    self.assertFalse(result["reject_primary"])
                    self.assertTrue(result["unavailable_components"])
                    json.dumps(result, allow_nan=False)

    def test_report_hierarchy_and_exact_four_claim_holm_mapping(self):
        cfg = an.Config(rounds=3, pairs=24, ri_draws=19)
        report = an.analyse(an.validate_rows(fixture_rows(24, 3), cfg), cfg)
        self.assertEqual(report["analysis_version"], "reference-3.0-primary-joint-signs")
        self.assertNotIn("primary_interaction", report)
        interaction = report["interaction_supporting"]
        self.assertNotIn("primary_p", interaction)
        self.assertNotIn("reject_primary", interaction)
        self.assertEqual(report["primary_joint_signs"], report["directional_sign_reversal_IUT"])
        raw = {"positive_endpoint_interaction": interaction.get("p_greater"),
               "gamma0_depth_ranking_equivalence": report["gamma0_depth_rank_TOST"]["p"],
               "conversion_ranking_equivalence_all_gamma": report["conversion_joint_TOST"]["p"],
               "near_CS_advantage_positive_gamma": report["near_CS_advantage_positive_gamma"].get("p_greater")}
        self.assertEqual(list(report["secondary_Holm_family"]), list(raw))
        self.assertEqual(report["secondary_Holm_family"], an.holm(raw, cfg.alpha))
        self.assertNotIn("directional_sign_reversal_IUT", report["secondary_Holm_family"])
        self.assertTrue(report["supplementary_sharp_path_RI"]["not_a_test_of_primary_joint_sign_null"])
        self.assertTrue(report["HT_depth_bounds"]["interaction"]["positive_interaction_alone_is_not_primary_reversal_support"])

    def test_all_missing_report_primary_is_unavailable_not_discovery(self):
        cfg = an.Config(rounds=3, pairs=24, ri_draws=19)
        data = an.validate_rows(fixture_rows(24, 3), cfg)
        data["T"][:] = np.nan
        data["observed_opens"][:] = 0
        report = an.analyse(data, cfg)
        self.assertEqual(report["primary_joint_signs"]["primary_p"], 1.)
        self.assertFalse(report["primary_joint_signs"]["reject_primary"])
        self.assertEqual(report["secondary_Holm_family"]["positive_endpoint_interaction"]["holm_p"], 1.)
        self.assertFalse(report["primary_joint_signs_bound_support"]["finite_Hoeffding_IUT_gate"])

    def test_probability_defaults_and_strict_validation(self):
        self.assertEqual(an.Config().gamma_probabilities, (.4, .2, .4))
        for probabilities in ((.4, .2), (.4, .2, .4, 0), (0, .5, .5), (-.1, .6, .5),
                              (.4, .2, .5), (True, .2, .4), (float("nan"), .2, .4),
                              (.4, float("inf"), .4), (".4", .2, .4), None):
            with self.subTest(probabilities=probabilities), self.assertRaises(an.ValidationError):
                an.Config(gamma_probabilities=probabilities).validate()
        an.Config(gamma_probabilities=(1/3, 1/3, 1/3)).validate()

    def test_nonuniform_complete_ht_equals_manual_pair_scores(self):
        cfg = an.Config(rounds=2, pairs=12, ri_draws=19)
        data = an.validate_rows(fixture_rows(12, 2), cfg)
        for name in ("interaction", "near_simple", "far_simple"):
            scores = []
            for pair in range(cfg.pairs):
                score = 0.
                for i in np.flatnonzero(data["pair"] == pair):
                    g = int(data["gamma"][i])
                    if g == 1:
                        continue
                    if name == "near_simple" and data["rank"][i] != 0:
                        continue
                    if name == "far_simple" and data["rank"][i] != 1:
                        continue
                    direction = (-1 if g == 0 else 1)
                    rank_sign = 2*data["rank"][i]-1 if name == "interaction" else 1
                    score += direction*rank_sign*data["T"][i]/(cfg.rounds*cfg.gamma_probabilities[g])
                scores.append(score)
            bound = an.ht_bounds(data, cfg, name)
            np.testing.assert_allclose(bound["exact_HT_completion_extrema"], [np.mean(scores)]*2)
            self.assertEqual(bound["pair_endpoint_score_support"], [-10., 10.])

    def test_nonuniform_missing_bounds_match_all_completions(self):
        cfg = an.Config(rounds=2, pairs=12, ri_draws=19, gamma_probabilities=(.3, .2, .5))
        data = an.validate_rows(fixture_rows(12, 2), cfg)
        index = [int(np.flatnonzero((data["gamma"] == g) & (data["rank"] == a))[0])
                 for g, a in ((0, 0), (2, 1))]
        data["T"][index] = np.nan
        data["observed_opens"][index] = [1, 2]
        for name in ("interaction", "near_simple", "far_simple"):
            values = []
            for first, second in itertools.product(range(1, 5), range(2, 5)):
                complete = copy.deepcopy(data)
                complete["T"][index] = [first, second]
                values.append(an.ht_bounds(complete, cfg, name)["exact_HT_completion_extrema"][0])
            bound = an.ht_bounds(data, cfg, name)
            np.testing.assert_allclose(bound["exact_HT_completion_extrema"], [min(values), max(values)])
            self.assertAlmostEqual(bound["pair_endpoint_score_support"][1], 4/.3)

    def test_bound_support_conjunction_never_uses_positive_interaction_alone(self):
        def result(near_hi, far_lo):
            bounds = {"near_simple": {"finite_Hoeffding_envelope": [-2, near_hi],
                                       "pair_t_Bonferroni_envelope": [-2, near_hi]},
                      "far_simple": {"finite_Hoeffding_envelope": [far_lo, 2],
                                      "pair_t_Bonferroni_envelope": [far_lo, 2]},
                      "interaction": {"positive_interaction_finite_Hoeffding_gate": True}}
            return an.joint_signs_bound_support(bounds)
        for near_hi, far_lo in ((.1, .2), (-.2, -.1), (0, .2), (-.2, 0)):
            value = result(near_hi, far_lo)
            self.assertFalse(value["finite_Hoeffding_IUT_gate"])
            self.assertFalse(value["approximate_pair_t_IUT_gate"])
        self.assertTrue(result(-.1, .1)["finite_Hoeffding_IUT_gate"])
        self.assertTrue(result(-.1, .1)["approximate_pair_t_IUT_gate"])
        self.assertFalse(result(-.1, .1)["positive_interaction_gate_suffices"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
