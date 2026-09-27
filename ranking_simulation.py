#!/usr/bin/env python3
"""Ordered search under seven ranking families, N=4,...,128."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

P_LOW, P_HIGH, A = 1.0, 2.0, 4.5
GAMMA_MAX = 0.8
SEED = 1710
MARKET_SIZES = (4, 8, 16, 32, 64, 128)
BASE_GAMMAS = np.linspace(0.0, GAMMA_MAX, 101)
FAMILIES = (
    ('nearest', 'Nearest-first'),
    ('distant_promotion', 'Distant seller promoted'),
    ('adjacent_swaps', 'Near-first adjacent swaps'),
    ('centre_out', 'Centre-out'),
    # Optional third CPA family: replace the preceding line with
    # ('block_promotion', 'Near-first block promotions'),
    ('far_interleaved', 'Far-first alternating'),
    ('local_decline', 'Locally steepest decline'),
    ('farthest', 'Farthest-first'),
)
CPA_FAMILIES = ('nearest', 'adjacent_swaps', 'block_promotion')

def primitives(n):
    distances = np.linspace(1.0, 3.0, n)
    rho = np.linspace(1.1, 1.6, n)
    costs = 0.5 * (rho - 1.0)**2
    return distances, rho, costs

def ranking_order(key, n):
    if key == 'nearest':
        order = list(range(n))
    elif key == 'farthest':
        order = list(range(n-1, -1, -1))
    elif key == 'adjacent_swaps':
        # One-based ranks (1,3,2,5,4,...). CPA on the equal distance grid;
        # its second cumulative advantage is zero when n >= 3.
        order = [0]
        for i in range(1,n,2):
            order.extend([i+1,i] if i+1 < n else [i])
    elif key == 'block_promotion':
        # First inspect ranks 1,2; in each following consecutive block of
        # at most three ranks, promote its farthest seller. On the equal
        # grid, the initial CPA buffer is 2h; a full nonterminal block has
        # increments (-2,+1,+2)h. Partial and terminal blocks also obey CPA.
        order = list(range(min(2,n)))
        for start in range(2,n,3):
            stop = min(start+3,n)
            order.extend([stop-1,*range(start,stop-1)])
    elif key == 'local_decline':
        # E = sum_i a_i*delta_i. Match increasing distances to increasing
        # coefficients to maximize E, hence minimize d E[T]/d gamma at 0.
        # This uses the calibration's costs, never prices or current gamma.
        _,rho,_ = primitives(n)
        w = np.r_[0.,(2.-rho[1:])**np.arange(n-1)]
        a = np.arange(n)*w-(np.cumsum(w[::-1])[::-1]-w)
        order = np.argsort(np.argsort(a,kind='stable'),kind='stable').tolist()
    elif key == 'distant_promotion':
        order = [0, n-1, *range(1,n-1)]
    elif key == 'near_promotion':
        order = [n-1, 0, *range(n-2,0,-1)]
    elif key == 'centre_out':
        order = sorted(range(n), key=lambda i: (abs(2*i-(n-1)),i))
    elif key in ('near_interleaved', 'far_interleaved'):
        order, lo, hi = [], 0, n-1
        near = key == 'near_interleaved'
        while lo <= hi:
            if near:
                order.append(lo)
                lo += 1
            else:
                order.append(hi)
                hi -= 1
            near = not near
    else:
        raise ValueError(key)
    assert sorted(order) == list(range(n))
    return np.array(order, dtype=int)

def psi(x, low=P_LOW, high=P_HIGH):
    x = np.asarray(x)
    z = np.clip(x-low, 0.0, high-low)
    return z*z/(2*(high-low)) + np.maximum(x-high, 0.0)

def reservation(costs, low=P_LOW, high=P_HIGH):
    c = np.asarray(costs)
    width = high-low
    return np.where(c <= width/2, low+np.sqrt(2*width*c), c+(low+high)/2)

def numerical_zero(scale):
    # Only roundoff-sized positive gains count as numerical equality. Grid
    # approximation is checked separately and is never hidden inside this rule.
    return 64 * np.finfo(float).eps * max(1.0, float(scale))

def primitive_at(x, grid, values, area):
    """Exact antiderivative of a linear interpolant, including partial cells."""
    x = np.clip(x, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, x, side='right')-1, 0, len(grid)-2)
    h = x-grid[j]
    slope = (values[j+1]-values[j])/(grid[j+1]-grid[j])
    return area[j] + h*values[j] + 0.5*h*h*slope

def solve_thresholds(alphas, costs, grid_points=6001, *, force_recursive=False,
                     low=P_LOW, high=P_HIGH):
    """m[t] is tested BEFORE paying costs[t] and inspecting position t+1."""
    alphas, costs = np.asarray(alphas,float), np.asarray(costs,float)
    n = len(alphas)
    assert n == len(costs) and np.all(costs > 0) and high > low
    assert np.all(np.isfinite(alphas)) and np.all(np.isfinite(costs))
    assert np.all(np.diff(costs) >= 0) and grid_points >= 2
    rho = reservation(costs, low, high)
    zero = numerical_zero(max(np.max(np.abs(alphas)),np.max(costs)))
    ceiling = max(0.0,float(np.max(alphas)-low))
    m, g0 = np.zeros(n), np.zeros(n)
    if ceiling == 0:
        return m, -costs.copy()
    base = np.linspace(0., ceiling, grid_points)
    grid, premium = base.copy(), np.zeros(grid_points)
    width = high-low
    for t in range(n-1,-1,-1):
        area = np.r_[0., np.cumsum(0.5*(premium[:-1]+premium[1:])*np.diff(grid))]
        lower, upper = alphas[t]-high, alphas[t]-low
        top = max(0., upper)
        integ_top = primitive_at(top,grid,premium,area)
        def gain(s):
            retained = np.clip((s-lower)/width,0.,1.)
            integ_lower = primitive_at(np.maximum(s,max(0.,lower)),grid,premium,area)
            future = retained*np.interp(s,grid,premium)
            future += np.maximum(integ_top-integ_lower,0.)/width
            return psi(alphas[t]-s,low,high) - costs[t] + future
        g0[t] = float(gain(0.))
        if g0[t] <= zero:
            # Monotonicity of G implies stopping at all nonnegative states.
            m[t] = 0.
            grid, premium = base.copy(), np.zeros_like(base)
            continue
        # If the next tail stops above its threshold, this one-shot root is
        # exact whenever it lies weakly above that threshold.
        myopic = alphas[t]-rho[t]
        next_threshold = m[t+1] if t+1 < n else 0.
        if myopic >= next_threshold and not force_recursive:
            root = myopic
        else:
            root = brentq(gain,0.,ceiling,xtol=1e-13,rtol=2e-14)
        m[t] = root
        new_grid = np.unique(np.r_[base,root])
        new_premium = np.maximum(gain(new_grid),0.)
        new_premium[new_grid >= root] = 0.
        grid, premium = new_grid,new_premium
    return m,g0

def depth_moments(alphas, thresholds, low=P_LOW, high=P_HIGH):
    """Analytically integrate prices for arbitrary recursive thresholds."""
    alphas, thresholds = np.asarray(alphas), np.asarray(thresholds)
    n, reach = len(alphas), np.zeros(len(alphas))
    for k in range(n):
        if thresholds[k] <= 0:
            break
        if k == 0:
            reach[k] = 1.
        else:
            ceiling = np.minimum.accumulate(thresholds[1:k+1][::-1])[::-1]
            survival = np.clip((high-alphas[:k]+ceiling)/(high-low),0.,1.)
            reach[k] = np.prod(survival)
    mean = float(reach.sum())
    second = float((reach*(2*np.arange(1,n+1)-1)).sum())
    assert np.all(np.diff(reach) <= 1e-12)
    return mean,max(0.,second-mean*mean),reach

def simulate_depths(prices, alphas, thresholds):
    """Common iid prices BY POSITION; recall uses the running maximum."""
    samples,n = prices.shape
    depth = np.zeros(samples,dtype=np.int16)
    state = np.zeros(samples)
    active = np.arange(samples)
    for t in range(n):
        if not len(active) or thresholds[t] <= 0:
            break
        active = active[state[active] < thresholds[t]]
        depth[active] += 1
        state[active] = np.maximum(state[active],alphas[t]-prices[active,t])
    return depth

def local_slope(delta,rho):
    k = np.arange(1,len(delta))
    w = (2.-rho[1:])**(k-1)
    return -float(np.sum(w*(k*delta[1:]-np.cumsum(delta)[:-1])))

def cumulative_proximity(delta):
    """All B_m, m=1,...,N-1; CPA requires every entry nonnegative."""
    delta = np.asarray(delta,float)
    remaining_min = np.minimum.accumulate(delta[:0:-1])[::-1]
    return np.cumsum(remaining_min-delta[:-1])

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

N_SIMULATIONS = 150_000
GRID_POINTS = 6001

def scalar_n3_benchmark(alphas,costs):
    """Independent N=3 Bellman benchmark using scalar adaptive quadrature."""
    from scipy.integrate import quad
    a,c = np.asarray(alphas),np.asarray(costs)
    top = max(0.,max(a)-1.)
    m2 = max(0., a[2]-reservation(c[2]))
    def v2(s):
        return s+max(0.,float(psi(a[2]-s))-c[2])
    def expectation(v,s,alpha,breaks):
        knots=[1.,2.,*[alpha-z for z in [s,*breaks] if 1.<alpha-z<2.]]
        knots=sorted(set(knots))
        return sum(quad(lambda p:v(max(s,alpha-p)),lo,hi,
                        epsabs=2e-11,epsrel=2e-11)[0]
                   for lo,hi in zip(knots[:-1],knots[1:]))
    def g1(s):
        return -c[1]+expectation(v2,s,a[1],[m2])-s
    m1=0. if g1(0.)<=1e-13 else brentq(g1,0.,top,xtol=1e-12)
    def v1(s):
        return s+max(0.,g1(s))
    def g0(s):
        return -c[0]+expectation(v1,s,a[0],[m1,m2])-s
    m0=0. if g0(0.)<=1e-13 else brentq(g0,0.,top,xtol=1e-12)
    return np.array([m0,m1,m2])

def run_tests():
    """Assertions target model boundaries and independent identities."""
    for n in range(2,130):
        for key,_ in FAMILIES:
            assert np.array_equal(np.sort(ranking_order(key,n)),np.arange(n))
    assert np.array_equal(ranking_order('centre_out',5),[2,1,3,0,4])
    assert np.array_equal(ranking_order('centre_out',6),[2,3,1,4,0,5])
    assert np.array_equal(ranking_order('adjacent_swaps',8),[0,2,1,4,3,6,5,7])
    assert np.array_equal(ranking_order('block_promotion',8),[0,1,4,2,3,7,5,6])
    assert np.array_equal(ranking_order('block_promotion',4),[0,1,3,2])
    assert len({tuple(ranking_order(key,4)) for key,_ in FAMILIES})==len(FAMILIES)
    # Exhaustive small-market comparison checks the local assignment direction.
    from itertools import permutations
    d4,r4,_=primitives(4)
    best=min(local_slope(d4[list(order)],r4) for order in permutations(range(4)))
    assert abs(local_slope(d4[ranking_order('local_decline',4)],r4)-best)<1e-12
    assert np.allclose(psi(reservation(np.array([.02,.5,1.]))),[.02,.5,1.])
    # Antiderivative test catches the old cumulative-interpolation error.
    x=np.array([0.,1.]);v=np.array([0.,2.]);area=np.array([0.,1.])
    assert abs(primitive_at(.25,x,v,area)-.25**2)<1e-15
    # Stop before inspection, stop at exact equality, and never restart.
    p=np.array([[1.5,1.25,1.]])
    assert simulate_depths(p,[2.,2.,2.],[0.,1.,1.])[0]==0
    assert simulate_depths(p,[2.,2.,2.],[1.,.5,1.])[0]==1
    assert simulate_depths(p,[2.,2.,2.],[1.,0.,1.])[0]==1
    assert simulate_depths(p,[2.,2.,2.],[2.,2.,2.])[0]==3
    # Perfect recall: the second offer cannot erase the first good offer.
    assert simulate_depths(np.array([[1.25,1.9,1.]]),[2.,2.,2.],
                           [1.,1.,.5])[0]==2
    max_reference_error=0.
    for gamma,wanted in [(.5,2.125),(.6,2.),(.7,2.)]:
        a=2.6-gamma*np.array([2.,1.5,1.]);c=np.array([.01,.02,.5])
        m,_=solve_thresholds(a,c)
        reference=scalar_n3_benchmark(a,c)
        max_reference_error=max(max_reference_error,float(max(abs(m-reference))))
        assert max(abs(m-reference))<2e-6
        assert abs(depth_moments(a,m)[0]-wanted)<1e-10
    # Two-seller formula, including both signs, under active tails.
    for gamma in (0.,.05,.1):
        for delta in (np.array([1.,2.]),np.array([2.,1.])):
            c=np.array([.01,.02]);a=3.-gamma*delta
            m,_=solve_thresholds(a,c)
            wanted=2.-np.clip(reservation(c[1])+gamma*(delta[1]-delta[0])-1.,0.,1.)
            assert abs(depth_moments(a,m)[0]-wanted)<1e-12
    # Independent piecewise-polynomial Bellman fixtures. These include
    # non-compliant tails, where a global one-shot formula would be wrong.
    fixtures = [
        ('near_interleaved',1.,
         [1.3,.7613474610717803,.8571428571428565,.2545263746663669,
          .4142857142857144,0.,0.,0.],1.5226949221435606),
        ('centre_out',1.,
         [.512964579719421,.5262015112855523,.5714285714285713,
          .5618628741977444,.7000000000000003,.6042902711439675,
          .8285714285714285,0.],4.321865956350103),
        ('farthest',.58,
         [.8192595196630595,.8392595196630596,.8768675376741758,
          .9271496799176241,.9851454738210951,1.047266140822131,
          1.1101901686783426,1.17],4.9529137959077705),
    ]
    d=np.linspace(1.,3.,8);rho=np.linspace(1.2,1.75,8);c=.5*(rho-1.)**2
    for key,gamma,reference,wanted in fixtures:
        a=3.5-gamma*d[ranking_order(key,8)]
        m,_=solve_thresholds(a,c)
        assert max(abs(m-reference))<2e-7
        assert abs(depth_moments(a,m)[0]-wanted)<5e-7
    for n in MARKET_SIZES:
        d,rho,c=primitives(n)
        prices=np.random.default_rng(np.random.SeedSequence([SEED,n,99])).uniform(1.,2.,(2048,n))
        for key,_ in FAMILIES:
            delta=d[ranking_order(key,n)]
            cpa=bool(np.all(cumulative_proximity(delta)>=-1e-12))
            assert cpa==((key in CPA_FAMILIES) or (key=='local_decline' and n==4)),(n,key,'CPA classification')
            m,_=solve_thresholds(A+np.zeros(n),c)
            assert max(abs(m-(A-rho)))<1e-12
            benchmark=sum((2.-rho)**np.arange(n))
            assert abs(depth_moments(A+np.zeros(n),m)[0]-benchmark)<1e-12
            h=1e-5
            means=[]
            for gamma in (0.,h,2*h):
                a=A-gamma*delta;m,_=solve_thresholds(a,c)
                means.append(depth_moments(a,m)[0])
            slope=(4*means[1]-means[2]-3*means[0])/(2*h)
            assert abs(slope-local_slope(delta,rho))<2e-5,(n,key,slope)
        # CPA is a stochastic comparison of every reach probability; do not
        # impose the stronger sorted-order pathwise claim on unsorted orders.
        for key in CPA_FAMILIES:
            delta=d[ranking_order(key,n)];previous=None
            assert np.all(cumulative_proximity(delta)>=-1e-12)
            for gamma in (0.,.1,.275,.3,.58,.8,1.,1.8,2.):
                a=A-gamma*delta;m,_=solve_thresholds(a,c)
                reach=depth_moments(a,m)[2]
                if previous is not None:
                    assert np.all(reach<=previous+2e-7),(n,key,gamma,'CPA reach')
                previous=reach
        for key,grid,sign in [('nearest',[0.,.2,.58,1.,1.8],-1),
                              ('farthest',[0.,.1,.3,.58],1)]:
            delta=d[ranking_order(key,n)];previous=None
            for gamma in grid:
                a=A-gamma*delta;m,_=solve_thresholds(a,c)
                depth=simulate_depths(prices,a,m)
                if previous is not None:
                    assert np.all(sign*(depth-previous)>=0),(n,key,gamma)
                previous=depth
        # D=0 including threshold-zero and nonparticipation boundaries.
        for gamma in (0.,1.,1.75,2.3,3.):
            a=np.full(n,A-gamma);m,_=solve_thresholds(a,c)
            expected=np.maximum(a-rho,0.)
            expected[expected<numerical_zero(A)]=0.
            assert max(abs(m-expected))<1e-12
    # Deterministic, identical random stream from root seed 1710.
    assert np.array_equal(np.random.default_rng(SEED).random(100),
                          np.random.default_rng(SEED).random(100))
    return {'independent_n3_max_threshold_error':max_reference_error,
            'zero_threshold_example':{'gamma':.6,'mean':2.0},
            'tests':'passed'}

def build_curve(n,key,grid_points=GRID_POINTS):
    d,rho,c=primitives(n);delta=d[ranking_order(key,n)]
    cache={}
    def solve(gamma):
        gamma=float(gamma)
        if gamma not in cache:
            a=A-gamma*delta;m,g=solve_thresholds(a,c,grid_points)
            mean,var,reach=depth_moments(a,m)
            cache[gamma]=(m,g,mean,var,reach)
        return cache[gamma]
    base=[solve(g) for g in BASE_GAMMAS]
    events=[]
    for j in range(len(BASE_GAMMAS)-1):
        left,right=float(BASE_GAMMAS[j]),float(BASE_GAMMAS[j+1])
        ml,gl,el,vl,pl=base[j]
        mr,gr,er,vr,pr=base[j+1]
        candidates=np.flatnonzero((ml>0)&(mr==0)&(pl>1e-9))
        for t in candidates:
            if gr[t]>numerical_zero(A):
                raise AssertionError('A threshold vanished at a strictly positive gain.')
            if gr[t]>=0:
                root=right
            else:
                root=brentq(lambda g:solve(g)[1][t],left,right,xtol=2e-12)
            # Solve at high resolution before locating a discontinuity. The
            # location is a numerical root, not a Monte Carlo participation rate.
            radius=2e-5
            def refined_gain(g):
                return solve_thresholds(A-g*delta,c,2*grid_points-1)[1][t]
            lo,hi=max(0.,root-radius),min(GAMMA_MAX,root+radius)
            if refined_gain(lo)>0 and refined_gain(hi)<0:
                root=brentq(refined_gain,lo,hi,xtol=2e-12)
            eps=1e-6
            ma,ga=solve_thresholds(A-(root-eps)*delta,c,2*grid_points-1)
            mb,gb=solve_thresholds(A-(root+eps)*delta,c,2*grid_points-1)
            before=depth_moments(A-(root-eps)*delta,ma)[0]
            after=depth_moments(A-(root+eps)*delta,mb)[0]
            if t==0 or before-after>1e-6:
                events.append({'date':int(t),'gamma':root,'left_mean':before,
                               'right_mean':after,'jump':before-after})
    entry=min([x['gamma'] for x in events if x['date']==0],default=float('inf'))
    events=[x for x in events if x['gamma']<=entry+1e-9]
    # Retain the coarse friction grid and add its midpoints. Exact events
    # receive one-sided nodes; solid lines will never bridge a discontinuity.
    gammas=set(float(x) for x in np.linspace(0.,GAMMA_MAX,201))
    for event in events:
        root=event['gamma']
        gammas.update([max(0.,root-1e-6),root,min(GAMMA_MAX,root+1e-6)])
        gammas.update(float(x) for x in np.linspace(max(0.,root-.01),min(GAMMA_MAX,root+.01),9))
    gammas=sorted(gammas)
    cells=[]
    for gamma in gammas:
        if gamma > entry+1e-9:
            # Expected depth remains zero after entry fails (monotone G_0(0)).
            if gamma not in BASE_GAMMAS:
                continue
            m=np.zeros(n);g=np.full(n,np.nan);mean=var=0.;reach=np.zeros(n)
        else:
            near_event=any(abs(gamma-e['gamma'])<=2e-6 for e in events)
            if near_event:
                m,g=solve_thresholds(A-gamma*delta,c,2*grid_points-1)
                # The located equality is assigned to stopping, including the
                # positive-probability zero-surplus state.
                for event in events:
                    if abs(gamma-event['gamma'])<=1e-10:
                        m[event['date']]=0.
                mean,var,reach=depth_moments(A-gamma*delta,m)
            else:
                m,g,mean,var,reach=solve(gamma)
        cells.append({'gamma':gamma,'thresholds':m,'exact_mean':mean,
                      'exact_variance':var,'reach':reach})
    return delta,rho,c,cells,events

def one_market(n,samples,grid_points):
    start=time.perf_counter()
    # A stream per N makes each panel reproducible independently of execution
    # order and worker count. Every stream has the single root seed 1710.
    rng=np.random.default_rng(np.random.SeedSequence([SEED,n]))
    prices=rng.uniform(1.,2.,size=(samples,n))
    rows=[];all_events=[];diagnostics=[]
    max_z=max_nested_shift=0.
    for key,label in FAMILIES:
        delta,rho,c,cells,events=build_curve(n,key,grid_points)
        for cell in cells:
            gamma=cell['gamma'];m=cell['thresholds']
            if m[0]<=0:
                mean=se=0.;half_mean=0.
            else:
                depths=simulate_depths(prices,A-gamma*delta,m)
                mean=float(depths.mean())
                se=float(depths.std(ddof=1)/np.sqrt(samples))
                half_mean=float(depths[:samples//2].mean())
                z=abs(mean-cell['exact_mean'])/se if se>0 else 0.
                max_z=max(max_z,z)
                max_nested_shift=max(max_nested_shift,abs(mean-half_mean))
                assert z<8.,(n,key,gamma,'MC vs integrated mean',z)
            rows.append({'N':n,'ranking_key':key,'ranking':label,'gamma':gamma,
                         'mean_depth':mean,'standard_error':se,
                         'ci_lower':max(0.,mean-1.959963984540054*se),
                         'ci_upper':min(n,mean+1.959963984540054*se),
                         'integrated_mean':cell['exact_mean'],
                         'integrated_variance':cell['exact_variance'],
                         'half_sample_mean':half_mean,
                         'local_derivative_at_zero':local_slope(delta,rho)})
        for event in events:
            all_events.append({'N':n,'ranking_key':key,**event})
        valid=[x for x in cells if x['exact_mean']>0]
        peak=max(valid,key=lambda x:x['exact_mean'])
        # Check state-grid error at the peak and several points spanning the
        # friction range. Discontinuity locations get separate one-sided checks.
        checks=[peak]+[min(valid,key=lambda x:abs(x['gamma']-g)) for g in (.2,.58,1.,1.8)]
        err=0.
        for cell in checks:
            gamma=cell['gamma']
            if any(abs(gamma-e['gamma'])<2e-6 for e in events):
                continue
            m,_=solve_thresholds(A-gamma*delta,c,2*grid_points-1)
            difference=abs(depth_moments(A-gamma*delta,m)[0]-cell['exact_mean'])
            err=max(err,difference)
        assert err<1e-4,(n,key,'state-grid error',err)
        # Half-step gamma grid around each numerical maximum: tests peak
        # magnitude/location without assuming global monotonicity of mixed orders.
        fine_peak=peak['exact_mean']
        fine_gamma=peak['gamma']
        for gamma in np.linspace(max(0.,peak['gamma']-.015),min(GAMMA_MAX,peak['gamma']+.015),13):
            m,_=solve_thresholds(A-gamma*delta,c,grid_points)
            val=depth_moments(A-gamma*delta,m)[0]
            if val>fine_peak:
                fine_peak,fine_gamma=val,float(gamma)
        diagnostics.append({'N':n,'ranking_key':key,'peak_mean':fine_peak,
                            'peak_gamma':fine_gamma,'grid_peak':peak['exact_mean'],
                            'max_grid_mean_error':err,
                            'participation_cutoff':next((e['gamma'] for e in events if e['date']==0),None)})
        print(f'N={n:3d}: {label} complete',flush=True)
    # A materially larger sample at selected informative regimes. Prices
    # from the same RNG extend the original sample to twice its size.
    extra=rng.uniform(1.,2.,size=(samples,n))
    enlarged=[]
    d,rho,c=primitives(n)
    checks=[('farthest',.58),('adjacent_swaps',.4),('local_decline',.1),
            ('distant_promotion',.12)]
    if any(key=='block_promotion' for key,_ in FAMILIES):
        checks.append(('block_promotion',.4))
    for key,gamma in checks:
        delta=d[ranking_order(key,n)];a=A-gamma*delta;m,_=solve_thresholds(a,c,grid_points)
        x=simulate_depths(prices,a,m);y=simulate_depths(extra,a,m)
        full=np.r_[x,y];mean=float(full.mean());se=float(full.std(ddof=1)/np.sqrt(len(full)))
        truth=depth_moments(a,m)[0]
        assert abs(mean-truth)<=8*se+1e-10
        enlarged.append({'ranking_key':key,'gamma':gamma,'base_mean':float(x.mean()),
                         'double_sample_mean':mean,'double_sample_se':se,'integrated_mean':truth})
    return {'N':n,'rows':rows,'events':all_events,'diagnostics':diagnostics,
            'max_mc_z':max_z,'max_nested_mean_shift':max_nested_shift,
            'larger_sample_checks':enlarged,'seconds':time.perf_counter()-start}

def make_figure(results,output_dir,paper_font=False):
    expected={key for key,_ in FAMILIES}
    for result in results:
        if {row['ranking_key'] for row in result['rows']} != expected:
            raise ValueError('Saved results use different ranking families; rerun the simulation.')
    import matplotlib
    matplotlib.use('pgf' if paper_font else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'serif','font.size':19,
                         'pgf.texsystem':'pdflatex','pgf.rcfonts':False,
                         'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage[scale=0.995,shrink=0.025em,stretch=.15em]{newtxtext}\DeclareEncodingSubset{TS1}{ntxtlf}{0}\usepackage{amsmath}\usepackage[scale=0.995,slantedGreek]{newtxmath}',
                         'axes.labelsize':20,'axes.titlesize':21,'xtick.labelsize':17.5,
                         'ytick.labelsize':17.5,'legend.fontsize':18.5,'pdf.fonttype':42,
                         'ps.fonttype':42,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':300})
    color_cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors={key:color_cycle[i] for i,(key,_) in enumerate(FAMILIES)}
    colors['nearest'],colors['distant_promotion']=colors['distant_promotion'],colors['nearest']
    colors['distant_promotion'],colors['centre_out']=colors['centre_out'],colors['distant_promotion']
    styles={key:'-' for key,_ in FAMILIES}
    styles.update(adjacent_swaps='--',block_promotion='-.')
    fig,axes=plt.subplots(2,3,figsize=(14,10.5),sharex=True,sharey=False)
    fig.subplots_adjust(left=.082,right=.985,bottom=.09,top=.755,hspace=.29,wspace=.15)
    # Match the panel scales used in the current manuscript figure.
    panel_ymax={4:4.5,8:7.5,16:11.5,32:16.,64:16.,128:16.}
    for ax,result in zip(axes.flat,results):
        n=result['N']
        for key,label in FAMILIES:
            rows=sorted([r for r in result['rows'] if r['ranking_key']==key],key=lambda r:r['gamma'])
            events=sorted([e for e in result['events'] if e['ranking_key']==key],key=lambda e:e['gamma'])
            entry=next((e['gamma'] for e in events if e['date']==0),float('inf'))
            bounds=[0.,*[e['gamma'] for e in events],GAMMA_MAX+1e-6]
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                if lo>=entry:
                    continue
                # New segment starts at the equality (stopping) value. Each
                # segment ends on the left of the next discontinuity.
                part=[r for r in rows if lo<=r['gamma']<hi and r['mean_depth']>0]
                if not part:
                    continue
                x=np.array([r['gamma'] for r in part]);y=np.array([r['mean_depth'] for r in part])
                # Retain all pointwise CIs in CSV/JSON; omit shading here.
                # Closely spaced threshold losses create very short branches.
                # A full-width stroke on those branches produces thick nubs
                # beside the thin vertical jump. Match the jump stroke there;
                # retain the actual locations and sizes of every discontinuity.
                short_branch = key == 'centre_out' and lo > 0. and hi-lo < .025
                ax.plot(x,y,color=colors[key],lw=1.05 if short_branch else 2.7,
                        linestyle=styles[key],
                        solid_capstyle='butt' if short_branch else 'projecting',zorder=3)
            for event in events:
                if event['jump']>.00001:
                    # An actual model discontinuity, not a diagonal link to
                    # the last grid point. Entry collapse uses a thin dash.
                    ax.plot([event['gamma']]*2,[event['right_mean'],event['left_mean']],
                            color=colors[key],lw=1.05,
                            linestyle=(0,(4,3)) if event['date']==0 else '-',
                            solid_capstyle='butt',dash_capstyle='butt',zorder=2)
        ax.set_title(rf'$N={n}$',pad=12)
        ax.set_xlim(0.,GAMMA_MAX);ax.set_ylim(0.,panel_ymax[n])
        ax.set_xticks(np.arange(0.,GAMMA_MAX+.001,.2))
        ystep=1 if n==4 else 3 if n==16 else 2
        ax.set_yticks(np.arange(0.,panel_ymax[n]+.001,ystep))
        ax.grid(True,color='.88',lw=.75)
    fig.supxlabel(r'Transport friction, $\gamma$',fontsize=21,y=.025)
    fig.supylabel(r'Expected search depth, $\mathbb{E}[T^*]$',fontsize=21,x=.009)
    handles=[Line2D([0],[0],color=colors[key],lw=3,linestyle=styles[key],label=label)
             for key,label in FAMILIES]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,
               frameon=False,columnspacing=1.1,handlelength=2.0)
    fig.text(.52,.832,r'$A=4.5$; reservation prices in $[1.1,1.6]$; all tails active',
             ha='center',fontsize=15.5,color='.35')
    # No timestamp metadata, so the fixed-seed outputs are reproducible.
    fig.savefig(output_dir/'simulation_lower_costs.pdf',bbox_inches='tight',
                metadata={'CreationDate':None,'ModDate':None,'Creator':'ranking_simulation.py'})
    if not paper_font:
        fig.savefig(output_dir/'simulation_lower_costs.png',bbox_inches='tight')
        plt.close(fig)
        return
    # Rasterise the same TeX-rendered PDF to preserve identical typography.
    import subprocess
    subprocess.run(['pdftoppm','-png','-singlefile','-r','300',
                    str(output_dir/'simulation_lower_costs.pdf'),
                    str(output_dir/'simulation_lower_costs')],check=True)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results/rankings')
    parser.add_argument('--samples',type=int,default=N_SIMULATIONS)
    parser.add_argument('--grid-points',type=int,default=GRID_POINTS)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--plot-only',action='store_true',help='Use saved simulation_lower_costs_results.json')
    parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--paper-font',action='store_true',help='Requires pdflatex, newtx and pdftoppm')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        payload=json.loads((args.output_dir/'simulation_lower_costs_results.json').read_text())
        make_figure(payload['results'],args.output_dir,args.paper_font);return
    tests=run_tests();print(json.dumps(tests),flush=True)
    if args.check_only:
        return
    assert args.samples>=2000 and args.grid_points>=1001
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one_market,n,args.samples,args.grid_points) for n in MARKET_SIZES]
        results=[future.result() for future in futures]
    payload={'seed':SEED,'samples':args.samples,'grid_points':args.grid_points,
             'ranking_families':[{'key':key,'label':label,
                                  'cpa_by_market':{str(n):bool(np.all(cumulative_proximity(
                                      primitives(n)[0][ranking_order(key,n)])>=-1e-12))
                                      for n in MARKET_SIZES}}
                                 for key,label in FAMILIES],
             'calibration':{'price_support':[1.,2.],'A':A,'distance_endpoints':[1.,3.],
                            'reservation_endpoints':[1.1,1.6],'gamma_range':[0.,GAMMA_MAX]},
             'tests':tests,'results':results}
    (args.output_dir/'simulation_lower_costs_results.json').write_text(json.dumps(payload,indent=2)+'\n')
    all_rows=[row for result in results for row in result['rows']]
    with (args.output_dir/'simulation_lower_costs_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    if not args.no_plot:
        make_figure(results,args.output_dir,args.paper_font)
    print('Saved simulation_lower_costs.pdf and simulation_lower_costs.png',flush=True)

if __name__=='__main__':
    main()
