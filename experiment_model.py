"""Exact finite-grid policies, expectations and realised outcomes."""
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    data: dict

    @classmethod
    def load(cls,path=None):
        path=Path(path) if path else Path(__file__).with_name('experiment_config.json')
        cfg=cls(json.loads(path.read_text()))
        assert cfg.data['n_sellers']==len(cfg.data['distances'])
        assert cfg.data['outside_step_points']==cfg.data['price_step_points']
        return cfg

    @property
    def unit(self):return float(self.data['price_step_points'])

    def units(self,points):
        scaled=Decimal(str(points))/Decimal(str(self.unit))
        if scaled!=scaled.to_integral_value():
            raise ValueError(f'{points} is not an exact multiple of {self.unit} points.')
        return int(scaled)

    @property
    def prices_units(self):
        return np.arange(self.units(self.data['price_min_points']),self.units(self.data['price_max_points'])+1,dtype=np.int64)

    @property
    def outside_units(self):
        return np.arange(self.units(self.data['outside_min_points']),self.units(self.data['outside_max_points'])+1,dtype=np.int64)

    @property
    def ranking_names(self):return tuple(self.data['rankings'])

    @property
    def cost_names(self):return tuple(self.data['inspection_costs_points'])

    def ordered_distances(self,ranking):
        order=np.asarray(self.data['rankings'][ranking],dtype=int)-1
        return np.asarray(self.data['distances'],dtype=float)[order]

    def costs_units(self,cost):
        return np.full(self.data['n_sellers'],self.units(self.data['inspection_costs_points'][cost]),dtype=np.int64)


@dataclass
class FinitePolicy:
    ranking: str
    cost: str
    gamma: float
    outside_units: int
    stop_state_units: np.ndarray
    continue_by_stage: np.ndarray
    gains_units: np.ndarray
    reward_probabilities: np.ndarray
    states_units: np.ndarray
    initial_value_units: float

    def continue_at(self,stage,state_units):
        """Decision before inspection stage+1; state is floored at zero upstream."""
        if stage<0 or stage>len(self.stop_state_units):raise ValueError('Invalid stage.')
        if not np.issubdtype(np.asarray(state_units).dtype,np.integer):raise ValueError('Use integer surplus units.')
        if np.any(np.asarray(state_units)<0):raise ValueError('Best surplus must be nonnegative.')
        if stage==len(self.stop_state_units):return np.zeros_like(state_units,dtype=bool)
        return np.asarray(state_units)<self.stop_state_units[stage]


class FiniteSolver:
    def __init__(self,config):
        self.config=config
        self.prices=config.prices_units

    def solve_policy(self,ranking,cost,gamma,outside_units):
        cfg=self.config;n=cfg.data['n_sellers'];outside_units=int(outside_units)
        d=cfg.ordered_distances(ranking)
        transport=np.array([cfg.units(float(gamma*di)) for di in d])
        A=cfg.units(cfg.data['value_points'])-outside_units
        costs=cfg.costs_units(cost)
        states=np.arange(max(0,A-int(min(transport))-int(self.prices[0]))+1,dtype=np.int64)
        probs=np.array([np.bincount(np.maximum(0,A-ti-self.prices),minlength=len(states))/len(self.prices) for ti in transport])
        values=states.astype(float)
        gains=np.zeros((n,len(states)))
        policies=np.zeros_like(gains,dtype=bool)
        for t in range(n-1,-1,-1):
            weighted=probs[t]*values
            tail=np.cumsum(weighted[::-1])[::-1]-weighted
            continuation=-costs[t]+values*np.cumsum(probs[t])+tail
            gains[t]=continuation-states
            # 1e-8 integer units equals 1e-10 points. Checked policy margins
            # exceed this numerical tolerance by more than five orders.
            policies[t]=gains[t]>1e-8
            values=np.maximum(states,continuation)
        stops=np.array([int(np.flatnonzero(~row)[0]) for row in policies],dtype=np.int64)
        assert all(np.array_equal(row,states<stop) for row,stop in zip(policies,stops))
        return FinitePolicy(ranking,cost,float(gamma),outside_units,stops,policies,gains,probs,states,float(values[0]))

    def expected_outcomes(self,policy):
        cfg=self.config;d=cfg.ordered_distances(policy.ranking);n=len(d)
        costs=cfg.costs_units(policy.cost);u=policy.outside_units
        A=cfg.units(cfg.data['value_points'])-u
        alive=np.zeros(len(policy.states_units));alive[0]=1
        dist_alive=np.zeros_like(alive);terminal=np.zeros_like(alive);dist_terminal=np.zeros_like(alive)
        reaches=[];search_ties=[];seller_ties=[]
        for t in range(n):
            cont=policy.continue_by_stage[t];probs=policy.reward_probabilities[t]
            search_ties.append(float(sum(alive[np.abs(policy.gains_units[t])<=1e-8])))
            terminal+=alive*(~cont);dist_terminal+=dist_alive*(~cont)
            alive=alive*cont;dist_alive=dist_alive*cont
            reaches.append(float(sum(alive)))
            cdf=np.cumsum(probs)
            improvements=probs*(np.cumsum(alive)-alive)
            seller_ties.append(float(sum((alive*probs)[1:])))
            dist_alive=dist_alive*cdf+improvements*d[t]
            alive=alive*cdf+improvements
        terminal+=alive;dist_terminal+=dist_alive
        q=float(sum(terminal[1:]));surplus=float(np.dot(policy.states_units,terminal))
        selected_distance=float(sum(dist_terminal[1:]))
        sc=float(np.dot(costs,reaches));tc=cfg.units(policy.gamma)*selected_distance
        revenue=A*q-surplus-tc;cs=u+surplus-sc
        ET=float(sum(reaches));sd=float(np.sqrt(max(0,np.dot(np.arange(1,n+1)*2-1,reaches)-ET*ET)))
        unit=cfg.unit
        assert abs(sum(terminal)-1)<1e-10
        assert abs(cs-(u+policy.initial_value_units))*unit<1e-8
        assert abs((cs+revenue)-(cfg.units(cfg.data['value_points'])*q+u*(1-q)-tc-sc))*unit<1e-8
        if np.all(policy.stop_state_units>0):
            assert abs(q-(1-np.prod(policy.reward_probabilities[:,0])))<1e-10
        return dict(depth=ET,depth_sd=sd,entry_probability=reaches[0],conversion=q,exit_probability=1-q,
                    continuation_after_first=ET-reaches[0],consumer_surplus=cs*unit,search_cost=sc*unit,
                    transport_cost=tc*unit,revenue=revenue*unit,welfare=(cs+revenue)*unit,
                    chosen_distance_unconditional=selected_distance,
                    chosen_distance_given_purchase=selected_distance/q if q else None,
                    price_given_purchase=revenue*unit/q if q else None,reach=reaches,
                    search_threshold_tie_mass=search_ties,positive_seller_tie_encounter_mass=seller_ties,
                    minimum_absolute_gain_points=float(np.min(abs(policy.gains_units)))*unit)

    def reservation_price(self,cost_points):
        """Exact root of the finite-grid expected-improvement equation."""
        p=self.prices.astype(float);k=np.arange(1,len(p)+1)
        roots=(len(p)*self.config.units(cost_points)+np.cumsum(p))/k
        upper=np.r_[p[1:],np.inf]
        eligible=np.flatnonzero((roots>=p)&(roots<=upper))
        assert len(eligible)>0
        return float(roots[eligible[0]]*self.config.unit)


class PolicyBank:
    """Compact exact stopping states; array order is ranking,cost,gamma,type,stage."""
    def __init__(self,config,stop_states):
        self.config=config;self.stop_states=np.asarray(stop_states,dtype=np.int64)

    @classmethod
    def build(cls,config):
        solver=FiniteSolver(config)
        shape=(len(config.ranking_names),len(config.cost_names),len(config.data['gammas']),len(config.outside_units),config.data['n_sellers'])
        cutoffs=np.empty(shape,dtype=np.int64)
        for ri,r in enumerate(config.ranking_names):
            for ci,c in enumerate(config.cost_names):
                for gi,g in enumerate(config.data['gammas']):
                    for ui,u in enumerate(config.outside_units):cutoffs[ri,ci,gi,ui]=solver.solve_policy(r,c,g,u).stop_state_units
        return cls(config,cutoffs)

    def save(self,path):
        np.savez_compressed(path,stop_states=self.stop_states,outside_units=self.config.outside_units,
                            gammas=np.asarray(self.config.data['gammas']),ranking_names=np.array(self.config.ranking_names),
                            cost_names=np.array(self.config.cost_names),money_unit_points=self.config.unit)

    @classmethod
    def load(cls,path,config=None):
        cfg=config or ModelConfig.load()
        with np.load(path,allow_pickle=False) as data:
            assert np.array_equal(data['outside_units'],cfg.outside_units)
            assert tuple(data['ranking_names'])==cfg.ranking_names
            assert tuple(data['cost_names'])==cfg.cost_names
            assert np.array_equal(data['gammas'],cfg.data['gammas'])
            assert float(data['money_unit_points'])==cfg.unit
            return cls(cfg,data['stop_states'])

    def thresholds(self,ranking,cost,gamma,outside_units):
        u=np.asarray(outside_units)
        if not np.issubdtype(u.dtype,np.integer):raise ValueError('Use integer outside-option units.')
        indices=u-int(self.config.outside_units[0])
        if np.any(indices<0) or np.any(indices>=len(self.config.outside_units)):raise ValueError('Outside option is outside configured support.')
        return self.stop_states[self.config.ranking_names.index(ranking),self.config.cost_names.index(cost),
                                self.config.data['gammas'].index(gamma),indices]

    def realised_policy(self,prices_units,outside_units,ranking,cost,gamma):
        """Apply rational policies to rows of SELLER-indexed integer prices."""
        cfg=self.config;p=np.asarray(prices_units)
        if p.ndim!=2 or p.shape[1]!=cfg.data['n_sellers'] or not np.issubdtype(p.dtype,np.integer):
            raise ValueError('prices_units must be an integer matrix with one column per seller.')
        if np.any(p<cfg.prices_units[0]) or np.any(p>cfg.prices_units[-1]):raise ValueError('Price outside configured support.')
        u=np.broadcast_to(np.asarray(outside_units),(len(p),))
        thresholds=self.thresholds(ranking,cost,gamma,u)
        order=np.asarray(cfg.data['rankings'][ranking])-1;d=cfg.ordered_distances(ranking)
        A=cfg.units(cfg.data['value_points'])-u
        s=np.zeros(len(p),dtype=np.int64);active=np.ones(len(p),dtype=bool)
        winner=np.zeros(len(p),dtype=np.int64);distance=np.zeros(len(p));depth=np.zeros(len(p),dtype=np.int64)
        for t,seller in enumerate(order):
            active=active&(s<thresholds[:,t]);depth+=active
            reward=A-p[:,seller]-cfg.units(float(gamma*d[t]))
            improve=active&(reward>s)
            s[improve]=reward[improve];winner[improve]=seller+1;distance[improve]=d[t]
        purchase=s>0;costpoints=cfg.data['inspection_costs_points'][cost]
        sc=depth*costpoints;tc=gamma*distance;revenue=np.where(purchase,(A-s)*cfg.unit-tc,0.)
        cs=u*cfg.unit+s*cfg.unit-sc;score=cfg.data['payment']['score_endowment_points']+cs
        return dict(depth=depth,conversion=purchase,selected_seller=winner,selected_distance=distance,
                    best_surplus_units=s,search_cost=sc,transport_cost=tc,revenue=revenue,
                    consumer_surplus=cs,welfare=cs+revenue,score=score)


def payment_summary(config,mean_consumer_surplus):
    p=config.data['payment'];mean_score=p['score_endowment_points']+mean_consumer_surplus
    return dict(mean_consumer_surplus_points=float(mean_consumer_surplus),mean_score_points=float(mean_score),
                prize_probability_full_completion=float(mean_score/p['score_denominator_points']),
                expected_payment_euros=float(p['fixed_euros']+p['prize_euros']*mean_score/p['score_denominator_points']))


def score_terminal(config,*,inspections,cost,gamma,outside_points,purchase_price_points=None,purchase_distance=None):
    """All-action payment score; permits inferior or negative-surplus purchases."""
    if not 0<=inspections<=config.data['n_sellers']:raise ValueError('Invalid inspection count.')
    c=config.data['inspection_costs_points'][cost]
    if purchase_price_points is None:
        y=outside_points-c*inspections
    else:
        if inspections<1 or purchase_distance is None:raise ValueError('A purchase requires an inspected seller.')
        y=config.data['value_points']-purchase_price_points-gamma*purchase_distance-c*inspections
    return float(config.data['payment']['score_endowment_points']+y)
