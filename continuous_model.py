"""Continuous-uniform reference solver, separate from the implemented grid."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial as P

SCALE = 100.0
X = P([0., 1.])


@dataclass
class Piecewise:
    breaks: list
    polys: list

    def polynomial(self, x):
        j = min(max(np.searchsorted(self.breaks, x, side='right') - 1, 0), len(self.polys)-1)
        return self.polys[j]

    def __call__(self, x):
        return float(self.polynomial(x)(x))

    def integral(self, a, b):
        if b < a:
            return -self.integral(b, a)
        total = 0.
        for l, r, p in zip(self.breaks[:-1], self.breaks[1:], self.polys):
            ll, rr = max(a, l), min(b, r)
            if ll < rr:
                ip = p.integ()
                total += ip(rr)-ip(ll)
        return float(total)


def bellman(costs, shifts, pmin=.2, pmax=1.2):
    n = len(costs)
    lo, hi = min(0., pmin+min(shifts)-1), pmax+max(shifts)+sum(costs)+2
    values = [None]*(n+1)
    values[n] = Piecewise([lo, hi], [X])
    cutoffs = np.zeros(n)
    width = pmax-pmin
    for t in range(n-1, -1, -1):
        nxt = values[t+1]
        a, b = pmin+shifts[t], pmax+shifts[t]
        cuts = sorted(set(nxt.breaks+[a,b]))
        polys = []
        for l, r in zip(cuts[:-1], cuts[1:]):
            mid = (l+r)/2
            poly = nxt.polynomial(mid)
            if mid < a:
                q = costs[t]+poly
            elif mid < b:
                ip = poly.integ()
                integral_poly = P([nxt.integral(a,l)-ip(l)])+ip
                q = costs[t]+(integral_poly+(b-X)*poly)/width
            else:
                q = P([costs[t]+nxt.integral(a,b)/width])
            polys.append(q)
        qfun = Piecewise(cuts, polys)
        left, right = lo, hi
        for _ in range(100):
            mid = (left+right)/2
            if mid-qfun(mid) > 0:
                right = mid
            else:
                left = mid
        root = (left+right)/2
        cutoffs[t] = root
        valuecuts = sorted(set(cuts+[root]))
        valuepolys = [X if (l+r)/2 < root else qfun.polynomial((l+r)/2)
                      for l,r in zip(valuecuts[:-1],valuecuts[1:])]
        values[t] = Piecewise(valuecuts,valuepolys)
    return cutoffs, values


def event_moments(lowers, uppers, winner, cap, width):
    """Unconditional mass and E[minimum * indicator], for one winner."""
    a, b = lowers[winner], min(uppers[winner], cap)
    if b <= a:
        return 0.,0.
    cuts = sorted(set([a,b]+[x for x in list(lowers)+list(uppers) if a<x<b]))
    prob, value = 0.,0.
    for l,r in zip(cuts[:-1],cuts[1:]):
        mid = (l+r)/2
        density = P([1/width])
        for j in range(len(lowers)):
            if j == winner:
                continue
            if mid >= uppers[j]:
                density = P([0.])
                break
            density *= (P([uppers[j]-lowers[j]]) if mid < lowers[j]
                        else P([uppers[j]])-X)/width
        ip = density.integ()
        iy = (density*X).integ()
        prob += ip(r)-ip(l)
        value += iy(r)-iy(l)
    return float(prob),float(value)


def calculate(v, u0, distances, gamma, costs, ranking, pmin=20, pmax=120):
    n=len(distances)
    d=np.asarray(distances)[np.asarray(ranking)]
    shifts=gamma*d/SCALE
    c=np.asarray(costs)/SCALE
    amin,amax=pmin/SCALE,pmax/SCALE
    width=amax-amin
    A=(v-u0)/SCALE
    rho=amin+np.sqrt(2*width*c)
    assert all(rho < amax), 'These examples assume interior reservation prices.'
    R,values=bellman(c,shifts,amin,amax)
    threshold=np.maximum(A-R,0)
    # All reporting/calibration families are strictly active.
    assert all(threshold>0), (ranking,gamma,costs,threshold)
    reaches=[]
    selected=np.zeros(n)
    revenue,tc=0.,0.
    for k in range(1,n+1):
        lowers=amin+shifts[:k].copy()
        uppers=amax+shifts[:k]
        for j in range(k-1):
            lowers[j]=max(lowers[j],max(R[j+1:k]))
        lowers=np.minimum(lowers,uppers)
        reach=np.prod((uppers[:-1]-lowers[:-1])/width)
        reaches.append(float(reach))
        stopcap=min(A,R[k]) if k<n else A
        for j in range(k):
            q,y=event_moments(lowers,uppers,j,stopcap,width)
            selected[j]+=q
            revenue+=y-shifts[j]*q
            tc+=shifts[j]*q
    conversion=float(sum(selected))
    exactconv=float(1-np.prod(1-np.clip((A-shifts-amin)/width,0,1)))
    search=float(np.dot(c,reaches))
    cs=u0+SCALE*(A*conversion-revenue-tc-search)
    dp_cs=u0+SCALE*(A-values[0](A))
    if abs(conversion-exactconv)>1e-8 or abs(cs-dp_cs)>1e-7:
        raise ValueError(('Analytical cross-check failed',conversion,exactconv,cs,dp_cs))
    depth=sum(reaches)
    return dict(ranking=[i+1 for i in ranking], gamma=gamma,
                costs=list(costs), distances=list(distances), v=v,u0=u0,
                rho=list(SCALE*rho),cutoffs=list(SCALE*R),
                thresholds=list(SCALE*threshold),reach=reaches,
                depth=depth,depth_sd=float(np.sqrt(np.dot(np.arange(1,n+1)*2-1,reaches)-depth**2)),
                conversion=conversion,search_cost=SCALE*search,
                transport_cost=SCALE*tc,revenue=SCALE*revenue,
                consumer_surplus=cs,welfare=cs+SCALE*revenue,
                selected_by_position=list(selected),
                active_margin=v-u0-gamma*max(distances)-SCALE*rho[-1],
                strict_margin=min(SCALE*rho[1]-pmin,pmax-SCALE*rho[1])-gamma*(max(distances)-min(distances)),
                compliance_margins=list(np.diff(SCALE*rho)-gamma*(d[:-1]-d[1:])))


def average_continuous(config, ranking, cost, gamma):
    """Integrate independent continuous U[49,51] types for the reference model."""
    data=config.data
    nodes,weights=np.polynomial.legendre.leggauss(5)
    lower=data['outside_min_points'];upper=data['outside_max_points']
    types=(upper+lower)/2+(upper-lower)/2*nodes
    weights=weights/2
    order=np.asarray(data['rankings'][ranking])-1
    costs=[data['inspection_costs_points'][cost]]*data['n_sellers']
    # Under this checked support, terminal-event expectations are polynomials
    # of degree at most five in u0; five-node Gauss integration is exact.
    acceptance_min=data['value_points']-upper-gamma*max(data['distances'])
    acceptance_max=data['value_points']-lower-gamma*min(data['distances'])
    if not (data['price_min_points']<acceptance_min<=acceptance_max<data['price_max_points']):
        raise ValueError('Continuous type integration requires acceptance support to stay inside price support; split integration intervals after changing this configuration.')
    rows=[calculate(data['value_points'],float(u),data['distances'],gamma,costs,order,
                    data['price_min_points'],data['price_max_points']) for u in types]
    keys=['depth','conversion','consumer_surplus','search_cost','transport_cost','revenue','welfare']
    result={key:float(sum(w*r[key] for w,r in zip(weights,rows))) for key in keys}
    distance=np.asarray(data['distances'])[order]
    selected=float(sum(w*np.dot(distance,r['selected_by_position']) for w,r in zip(weights,rows)))
    result.update(entry_probability=1.0,exit_probability=1-result['conversion'],
                  continuation_after_first=result['depth']-1,
                  chosen_distance_unconditional=selected,
                  chosen_distance_given_purchase=selected/result['conversion'],
                  price_given_purchase=result['revenue']/result['conversion'])
    centre=calculate(data['value_points'],(upper+lower)/2,data['distances'],gamma,costs,order,
                     data['price_min_points'],data['price_max_points'])
    result['thresholds_at_midpoint_type']=centre['thresholds']
    result['generalised_price_cutoffs']=centre['cutoffs']
    result['rho']=centre['rho']
    result['reach']=centre['reach']
    return result
