"""Deterministic verification of final predictions and the realised-policy API."""
from copy import deepcopy
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import unittest
import numpy as np
from experiment.experiment_model import ModelConfig,FiniteSolver,PolicyBank,score_terminal,payment_summary
from experiment.experiment_predictions import assignment_mean_consumer_surplus

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'results/calibration'


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=ModelConfig.load()
        cls.solver=FiniteSolver(cls.cfg)
        cls.bank=PolicyBank.load(OUT/'policy_thresholds.npz',cls.cfg)
        cls.cells=json.loads((OUT/'finite_cells.json').read_text())

    def test_frozen_independent_prior_results(self):
        prior=json.loads((HERE/'fixtures/prior_validation_fixture.json').read_text())['cells']
        errors=[]
        for reference in prior:
            row=next(x for x in self.cells if all(x[k]==reference[k] for k in ['ranking','cost_label','gamma']))
            for key in reference:
                if key not in ['ranking','cost_label','gamma']:
                    errors.append(abs(row[key]-reference[key]))
        self.assertLess(max(errors),1e-8)
        self.__class__.prior_maximum_error=max(errors)

    def test_support_strictness_and_type_invariance(self):
        for row in self.cells:
            self.assertGreater(row['continuous_activity_margin'],0)
            self.assertGreater(row['worst_type_activity_margin_finite'],0)
            self.assertGreater(row['continuous_strictness_margin'],0)
            self.assertEqual(row['max_search_threshold_tie_mass'],0)
            self.assertLess(np.ptp(row['depth_range_over_types']),1e-10)
        self.assertTrue(np.all(self.bank.stop_states>0))
        # Every one-cent increase in u0 lowers the positive surplus stopping
        # states by exactly one integer unit throughout the configured support.
        self.assertTrue(np.all(np.diff(self.bank.stop_states,axis=3)==-1))
        for c in self.cfg.cost_names:
            for r in self.cfg.ranking_names:
                ordered=sorted([x for x in self.cells if x['cost_label']==c and x['ranking']==r],key=lambda x:x['gamma'])
                changes=np.diff([x['depth'] for x in ordered])
                self.assertTrue(np.all(changes<0) if r=='nearest' else np.all(changes>0))

    def test_conversion_product_independent_of_stopping(self):
        p=self.cfg.prices_units;v=self.cfg.units(self.cfg.data['value_points'])
        for gamma in self.cfg.data['gammas']:
            expected=[]
            for u in self.cfg.outside_units:
                failure=[np.mean(p>=v-u-self.cfg.units(float(gamma*d))) for d in self.cfg.data['distances']]
                expected.append(1-np.prod(failure))
            q=float(np.mean(expected))
            for row in self.cells:
                if row['gamma']==gamma:self.assertAlmostEqual(row['conversion'],q,places=11)

    def test_finite_reservation_equation(self):
        prices=self.cfg.prices_units*self.cfg.unit
        for fee in self.cfg.data['inspection_costs_points'].values():
            rho=self.solver.reservation_price(fee)
            self.assertAlmostEqual(float(np.mean(np.maximum(rho-prices,0))),fee,places=11)

    def test_positive_terminal_ties_keep_earliest_inspected(self):
        # The first two inspected sellers tie at positive surplus 10 and the
        # other prices are unaffordable. Both tied offers are actually reached.
        near=np.array([[5600,5200,12000,12000]],dtype=np.int64)
        far=np.array([[12000,12000,4800,4400]],dtype=np.int64)
        for prices,ranking,winner,distance in [(near,'nearest',1,1),(far,'farthest',4,4)]:
            result=self.bank.realised_policy(prices,np.array([5000]),ranking,'low',4)
            self.assertEqual(int(result['depth'][0]),4)
            self.assertEqual(int(result['selected_seller'][0]),winner)
            self.assertEqual(float(result['selected_distance'][0]),distance)
            self.assertEqual(int(result['best_surplus_units'][0]),1000)

    def test_zero_surplus_terminal_ties_exit(self):
        prices=np.array([[6600,6200,5800,5400]],dtype=np.int64)
        for ranking in self.cfg.ranking_names:
            result=self.bank.realised_policy(prices,np.array([5000]),ranking,'high',4)
            self.assertEqual(int(result['depth'][0]),4)
            self.assertFalse(bool(result['conversion'][0]))
            self.assertEqual(int(result['selected_seller'][0]),0)
            self.assertEqual(float(result['transport_cost'][0]),0)
            self.assertEqual(float(result['consumer_surplus'][0]),34)

    def test_outside_state_floor_and_inactive_policy(self):
        # Outside the study calibration, the floor must stop entry rather than
        # mechanically continue at m*=s=0 when continuing is strictly inferior.
        policy=self.solver.solve_policy('nearest','high',4,self.cfg.units(100))
        self.assertTrue(np.all(policy.stop_state_units==0))
        self.assertFalse(bool(policy.continue_at(0,0)))
        result=self.solver.expected_outcomes(policy)
        self.assertEqual(result['depth'],0)
        self.assertEqual(result['conversion'],0)
        self.assertEqual(result['consumer_surplus'],100)

    def test_payment_range_and_expected_payment(self):
        low=score_terminal(self.cfg,inspections=4,cost='high',gamma=4,outside_points=49,
                           purchase_price_points=120,purchase_distance=4)
        high=score_terminal(self.cfg,inspections=1,cost='low',gamma=0,outside_points=51,
                            purchase_price_points=20,purchase_distance=1)
        self.assertEqual(low,8);self.assertEqual(high,139)
        summary=payment_summary(self.cfg,np.mean([r['consumer_surplus'] for r in self.cells]))
        self.assertAlmostEqual(summary['expected_payment_euros'],9.50856332419536,places=10)
        self.assertGreaterEqual(summary['prize_probability_full_completion'],0)
        self.assertLessEqual(summary['prize_probability_full_completion'],1)

    def test_registered_assignment_weighted_payment(self):
        mean=assignment_mean_consumer_surplus(self.cfg,self.cells)
        summary=payment_summary(self.cfg,mean)
        self.assertAlmostEqual(300*summary['expected_payment_euros'],2853.031742089222,places=8)
        actual=json.loads((OUT/'payment_summary.json').read_text())
        self.assertEqual(actual['gamma_probabilities'],[.4,.2,.4])
        self.assertAlmostEqual(actual['expected_payment_euros'],summary['expected_payment_euros'],places=12)
        for probs in ([1,1,1],[.4,0,.6],[.4,.6]):
            cfg=ModelConfig(dict(self.cfg.data,gamma_probabilities=probs))
            with self.subTest(probabilities=probs),self.assertRaises(ValueError):
                assignment_mean_consumer_surplus(cfg,self.cells)

    def test_independent_brute_two_sellers(self):
        data=deepcopy(self.cfg.data)
        data.update(n_sellers=2,distances=[1,4],price_step_points=1,outside_step_points=1,
                    rankings={'nearest':[1,2],'farthest':[2,1]})
        cfg=ModelConfig(data);solver=FiniteSolver(cfg);bank=PolicyBank.build(cfg)
        prices=np.arange(20,121,dtype=np.int64)
        pairs=np.asarray(list(product(prices,repeat=2)))
        maxerr=0.
        for ranking in cfg.ranking_names:
            order=np.asarray(cfg.data['rankings'][ranking])-1
            d=cfg.ordered_distances(ranking);gamma=4;A=70
            for cost,fee in cfg.data['inspection_costs_points'].items():
                @lru_cache(None)
                def value(t,s):
                    if t==2:return s
                    return max(s,-fee+sum(value(t+1,max(s,A-int(p)-gamma*d[t])) for p in prices)/len(prices))
                @lru_cache(None)
                def cont(t,s):
                    return t<2 and -fee+sum(value(t+1,max(s,A-int(p)-gamma*d[t])) for p in prices)/len(prices)>s+1e-11
                rows=[]
                for pair in pairs:
                    s=0;winner=None;depth=0
                    for t,seller in enumerate(order):
                        if not cont(t,s):break
                        depth+=1;reward=A-int(pair[seller])-gamma*d[t]
                        if reward>s:s=reward;winner=seller
                    distance=0 if winner is None else cfg.data['distances'][winner]
                    revenue=0 if winner is None else pair[winner]
                    cs=50+s-fee*depth
                    rows.append([depth,float(winner is not None),cs,fee*depth,gamma*distance,revenue,cs+revenue,distance])
                keys=['depth','conversion','consumer_surplus','search_cost','transport_cost','revenue','welfare','chosen_distance_unconditional']
                averages=dict(zip(keys,np.mean(rows,axis=0)))
                expected=solver.expected_outcomes(solver.solve_policy(ranking,cost,gamma,50))
                realised=bank.realised_policy(pairs,np.full(len(pairs),50,dtype=np.int64),ranking,cost,gamma)
                for key in keys:
                    maxerr=max(maxerr,abs(expected[key]-averages[key]))
                    realised_key='selected_distance' if key=='chosen_distance_unconditional' else key
                    self.assertAlmostEqual(float(np.mean(realised[realised_key])),averages[key],places=10)
        self.assertLess(maxerr,1e-9)
        self.__class__.brute_maximum_error=maxerr


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PredictionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(tests_run=result.testsRun,passed=result.wasSuccessful(),
                frozen_prior_maximum_error=getattr(PredictionTests,'prior_maximum_error',None),
                independent_brute_maximum_error=getattr(PredictionTests,'brute_maximum_error',None))
    (OUT/'test_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)
