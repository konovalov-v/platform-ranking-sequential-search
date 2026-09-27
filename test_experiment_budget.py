"""Synthetic checks for the new weighted allocation; no old files are written."""
from fractions import Fraction
import unittest

import numpy as np

import experiment_budget as b


class EndpointBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.means, cls.policy_count, cls.exact = b.exact_type_means()

    def test_selected_allocation_and_unchanged_payment_bounds(self):
        np.testing.assert_array_equal(b.GAMMA_PROBABILITIES, [.4, .2, .4])
        self.assertEqual((b.J, b.STUDY_PAIRS, b.BASE_CENTS, b.PRIZE_CENTS, b.M_H), (24, 150, 300, 1200, 20000))
        self.assertEqual(2 * b.STUDY_PAIRS * (b.BASE_CENTS+b.PRIZE_CENTS) / 100, 4500)

    def test_independent_rational_weighted_expectation(self):
        r = self.exact["expected_total_euros"]
        value = Fraction(int(r["numerator"]), int(r["denominator"]))
        self.assertEqual(value, Fraction(28541731259793744985737, 10004000600040001000))
        self.assertAlmostEqual(float(value), 2853.031742089222, places=10)
        self.assertGreater(float(value)-2852.5689972586093, .46)

    def test_all_weighted_policy_means_match_rational_expectation(self):
        self.assertEqual(self.policy_count, 2412)
        expected = 900 + 1800 * (self.means.mean(axis=1) / 20000).sum()
        self.assertAlmostEqual(expected, self.exact["expected_total_euros"]["decimal"], places=8)

    def test_realised_scores_and_depths_match_saved_policy(self):
        checks = b.validate_policy(2026091428)
        self.assertEqual(checks["realised_score_and_depth_checks"], 2412)
        self.assertTrue(checks["convolution_two_pair_polynomial_verified"])

    def test_covariance_changes_variance_not_mean(self):
        mu = self.means.mean(axis=1)/20000
        independent, _ = b.distribution_summary(b.pair_probabilities(mu, 0))
        dependent, _ = b.distribution_summary(b.pair_probabilities(mu, .0002678))
        self.assertAlmostEqual(independent["expected_total_euros"], dependent["expected_total_euros"], places=9)
        self.assertGreater(dependent["standard_deviation_total_euros"], independent["standard_deviation_total_euros"])

    def test_strict_threshold_and_convolution_support(self):
        _, mass = b.distribution_summary(np.array([.2, .5, .3]))
        cash = 900 + 12*np.arange(len(mass))
        self.assertEqual(len(mass), 301)
        self.assertAlmostEqual(mass.sum(), 1, places=12)
        self.assertEqual(cash[175], 3000)
        self.assertEqual(np.flatnonzero(cash > 3000)[0], 176)
        self.assertEqual(cash[-1], 4500)

    def test_categorical_pair_stream_is_reproducible(self):
        first = b.simulate_batch(np.random.SeedSequence(1710), 100, 20, self.means)
        second = b.simulate_batch(np.random.SeedSequence(1710), 100, 20, self.means)
        self.assertEqual(first, second)
        self.assertEqual(sum(first["gamma_draw_counts"]), 100*24)
        self.assertTrue(all(count > 0 for count in first["gamma_draw_counts"]))
        self.assertAlmostEqual(sum(first["raw_conditional_pair_probabilities"]), 1, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
