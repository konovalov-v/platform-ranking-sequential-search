"""No-subject checks of assignment; run with python3 -m unittest test_randomisation."""
import copy
import hashlib
import unittest
from experiment.randomisation import CONFIG, Draws, audit, friction_from_ticket, generate, validate


class RandomisationTests(unittest.TestCase):
    def setUp(self):
        self.key = hashlib.sha256(b"test fixture, never a study key").digest()
        self.config = dict(CONFIG, pairs=12, rounds=24)

    def test_reproducibility_and_commitment(self):
        a = generate(self.key, self.config)
        self.assertEqual(a, generate(self.key, self.config))
        b = generate(hashlib.sha256(self.key).digest(), self.config)
        self.assertNotEqual(a["seed_commitment_sha256"], b["seed_commitment_sha256"])
        self.assertNotEqual(a["pairs"], b["pairs"])

    def test_paired_common_profiles_and_units(self):
        m = generate(self.key, self.config)
        self.assertTrue(validate(m))
        self.assertEqual(sum(audit(m)["pair_round_counts"].values()), 288)
        for pair in m["pairs"]:
            self.assertEqual({s["ranking"] for s in pair["slots"]}, {0, 1})
            for row in pair["rounds"]:
                near = row["seller_prices_hundredths"]
                far = list(reversed(near))
                self.assertEqual(sorted(near), sorted(far))

    def test_duplicate_domain_rejected(self):
        draws = Draws(self.key)
        draws.integer("x", 0, 2)
        with self.assertRaises(ValueError):
            draws.integer("x", 0, 2)

    def test_bad_key_and_bad_assignment_rejected(self):
        with self.assertRaises(ValueError):
            Draws(b"short")
        m = generate(self.key, self.config)
        bad = copy.deepcopy(m)
        bad["pairs"][0]["slots"][1]["ranking"] = bad["pairs"][0]["slots"][0]["ranking"]
        with self.assertRaises(ValueError):
            validate(bad)

    def test_no_imposed_personal_cell_balance(self):
        m = generate(self.key, dict(self.config, rounds=1))
        self.assertEqual(audit(m)["pairs_without_all_six_cells"], self.config["pairs"])
        self.assertTrue(validate(m))

    def test_practice_fixed_order_and_unpaid(self):
        m = generate(self.key, self.config)
        for pair in m["pairs"]:
            self.assertEqual([(r["cost"], r["gamma"]) for r in pair["practice"]],
                             [(0, 0), (1, 0), (0, 4), (1, 4)])
            self.assertTrue(all(r["paid_eligible"] is False for r in pair["practice"]))
        for defect in ("count", "order", "paid", "price"):
            bad = copy.deepcopy(m)
            practice = bad["pairs"][0]["practice"]
            if defect == "count":
                practice.pop()
            elif defect == "order":
                practice[0]["gamma"] = 2
            elif defect == "paid":
                practice[0]["paid_eligible"] = True
            else:
                practice[0]["seller_prices_hundredths"][0] = 1234
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate(bad)

    def test_non_integer_points_and_duplicate_pair_rejected(self):
        m = generate(self.key, self.config)
        for defect in ("float", "boolean", "pair_id"):
            bad = copy.deepcopy(m)
            if defect == "pair_id":
                bad["pairs"][1]["pair_id"] = bad["pairs"][0]["pair_id"]
            else:
                bad["pairs"][0]["rounds"][0]["seller_prices_hundredths"][0] = (
                    2500.5 if defect == "float" else True)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate(bad)

    def test_economic_configuration_change_rejected(self):
        bad = dict(self.config, gammas=[0, 2, 5])
        with self.assertRaises(ValueError):
            generate(self.key, bad)

    def test_exact_friction_probability_mapping(self):
        self.assertEqual([friction_from_ticket(k, self.config) for k in range(5)],
                         [0, 0, 2, 4, 4])
        for ticket in (-1, 5, 1.5, True):
            with self.subTest(ticket=ticket), self.assertRaises(ValueError):
                friction_from_ticket(ticket, self.config)

    def test_probability_change_is_a_protocol_change(self):
        for weights in ([1, 1, 1], [4, 2, 4], [2, 0, 3]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                generate(self.key, dict(self.config, gamma_probability_weights=weights))


if __name__ == "__main__":
    unittest.main()
