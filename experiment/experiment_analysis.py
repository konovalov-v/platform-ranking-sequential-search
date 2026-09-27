"""Reference analysis for the fixed-enrolment paired iid search experiment."""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


GAMMAS = (0, 2, 4)
ANALYSIS_VERSION = "reference-3.0-primary-joint-signs"
NUMERICAL_SE_REL_TOLERANCE = 1e-12
INDUCED_CS_SUPPORT = (-32.0, 99.0)
FEE_LEVELS = (1.0, 4.0)
OPTIONAL_CHOICE_SUPPORTS = {"TC": (0.0, 16.0), "REV": (0.0, 120.0), "W": (33.0, 119.0)}
IDENTITY_TOLERANCE = 1e-8
REQUIRED_COLUMNS = (
    "pair_id", "participant_id", "rank", "round", "cost", "gamma", "T",
    "depth_observed", "terminal_complete", "observed_opens", "purchase", "induced_CS",
)
MISSING_TOKENS = {"", "na", "nan", "null", "none"}


class ValidationError(ValueError):
    """Input contradicts the registered all-assigned-slot design."""


class EstimationError(ValueError):
    """A specified point contrast or its inference cannot be calculated."""


@dataclass(frozen=True)
class Config:
    rounds: int = 24
    pairs: int = 150
    alpha: float = 0.05
    depth_equivalence_margin: float = 0.25
    conversion_equivalence_margin: float = 0.05
    ri_draws: int = 9999
    ri_seed: int = 2026091324
    gamma_probabilities: tuple = (0.4, 0.2, 0.4)

    def validate(self):
        if self.rounds < 1 or self.pairs < 2:
            raise ValidationError("Positive rounds and at least two assigned pairs required")
        if not 0 < self.alpha < 0.5:
            raise ValidationError("alpha must lie between 0 and 0.5")
        if self.depth_equivalence_margin <= 0 or self.conversion_equivalence_margin <= 0:
            raise ValidationError("Equivalence margins must be positive")
        if self.ri_draws < 1:
            raise ValidationError("At least one RI draw required")
        if (not isinstance(self.gamma_probabilities, (tuple, list)) or
                len(self.gamma_probabilities) != 3 or
                any(isinstance(p, bool) or not isinstance(p, (int, float)) or
                    not math.isfinite(p) or p <= 0 for p in self.gamma_probabilities) or
                not math.isclose(sum(self.gamma_probabilities), 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ValidationError("Three strictly positive finite gamma probabilities summing to one are required")


def _number(value, field, line, missing=False, integer=False):
    value = str(value).strip()
    if value.lower() in MISSING_TOKENS:
        if missing:
            return float("nan")
        raise ValidationError(f"Row {line}: {field} may not be missing")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValidationError(f"Row {line}: invalid {field}") from exc
    if not math.isfinite(result):
        raise ValidationError(f"Row {line}: {field} must be finite or explicitly missing")
    if integer:
        if not result.is_integer():
            raise ValidationError(f"Row {line}: {field} must be an integer")
        return int(result)
    return result


def _boolean(value, line, field="terminal_complete"):
    value = str(value).strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValidationError(f"Row {line}: {field} must be true/false or 1/0")


def validate_rows(rows, config):
    """Parse row dictionaries and verify all planned slots; never filter pairs."""
    config.validate()
    parsed = []
    participants = {}
    pairs = {}
    unique_slots = set()
    optional_outcomes_provided = set()
    rendered_columns_provided = False
    actual_final_column_provided = False
    for line, row in enumerate(rows, start=2):
        if None in row:
            raise ValidationError(f"Row {line}: more CSV fields than column headers")
        missing_columns = set(REQUIRED_COLUMNS)-set(row)
        if missing_columns:
            raise ValidationError(f"Missing columns: {sorted(missing_columns)}")
        pair = "" if row["pair_id"] is None else str(row["pair_id"]).strip()
        person = "" if row["participant_id"] is None else str(row["participant_id"]).strip()
        if not pair or not person:
            raise ValidationError(f"Row {line}: pair and participant IDs must be nonempty")
        a = _number(row["rank"], "rank", line, integer=True)
        r = _number(row["round"], "round", line, integer=True)
        c = _number(row["cost"], "cost", line, integer=True)
        g = _number(row["gamma"], "gamma", line, integer=True)
        if a not in (0, 1) or c not in (0, 1) or g not in GAMMAS:
            raise ValidationError(f"Row {line}: expected rank/cost 0/1 and gamma 0/2/4")
        if not 1 <= r <= config.rounds:
            raise ValidationError(f"Row {line}: round is outside configured planned rounds")
        complete = _boolean(row["terminal_complete"], line)
        depth_observed = _boolean(row["depth_observed"], line, "depth_observed")
        depth = _number(row["T"], "T", line, missing=True)
        opens = _number(row["observed_opens"], "observed_opens", line, missing=True)
        purchase = _number(row["purchase"], "purchase", line, missing=True)
        cs = _number(row["induced_CS"], "induced_CS", line, missing=True)
        actual_final = _number(row.get("actual_final_T", ""), "actual_final_T", line, missing=True)
        actual_final_column_provided |= "actual_final_T" in row
        optional = {name: _number(row.get(name, ""), name, line, missing=True)
                    for name in ("entry", "SC", "TC", "REV", "W")}
        rendered = _number(row.get("rendered_T", ""), "rendered_T", line, missing=True)
        rendered_flag = str(row.get("rendered_depth_observed", "")).strip()
        rendered_observed = (False if rendered_flag.lower() in MISSING_TOKENS
                             else _boolean(rendered_flag, line, "rendered_depth_observed"))
        rendered_columns_provided |= ("rendered_T" in row or "rendered_depth_observed" in row)
        optional_outcomes_provided.update(name for name in OPTIONAL_CHOICE_SUPPORTS if name in row)
        if math.isnan(opens):
            opens = 0.0  # Support lower bound, NOT an observed outcome.
        if not opens.is_integer() or not 0 <= opens <= 4:
            raise ValidationError(f"Row {line}: observed_opens must be an integer from 0 to 4")
        if math.isfinite(depth):
            if not depth.is_integer() or not 0 <= depth <= 4:
                raise ValidationError(f"Row {line}: T must be an integer from 0 to 4")
            if not depth_observed:
                raise ValidationError(f"Row {line}: T supplied without certified final depth_observed")
            if opens > depth:
                raise ValidationError(f"Row {line}: observed_opens exceeds terminal T")
            if not complete and (depth != 4 or opens != 4):
                raise ValidationError(f"Row {line}: completed-task T without an economic terminal choice requires T=4 and verified observed_opens=4; preserve partial administrative depth only as actual_final_T")
        elif depth_observed:
            raise ValidationError(f"Row {line}: depth_observed requires exact final T")
        if math.isfinite(actual_final):
            if not actual_final.is_integer() or not 0 <= actual_final <= 4:
                raise ValidationError(f"Row {line}: actual_final_T must be an integer from 0 to 4")
            if actual_final < opens:
                raise ValidationError(f"Row {line}: actual_final_T cannot be below verified observed_opens")
            if math.isfinite(depth) and actual_final != depth:
                raise ValidationError(f"Row {line}: certified completed-task T and actual_final_T disagree")
        if math.isfinite(rendered):
            if not rendered_observed:
                raise ValidationError(f"Row {line}: rendered_T requires certified rendered_depth_observed")
            if not rendered.is_integer() or not 0 <= rendered <= 4:
                raise ValidationError(f"Row {line}: rendered_T must be an integer from 0 to 4")
            if math.isfinite(depth) and rendered > depth:
                raise ValidationError(f"Row {line}: exact rendered_T cannot exceed exact committed T")
            if math.isfinite(actual_final) and rendered > actual_final:
                raise ValidationError(f"Row {line}: exact rendered_T cannot exceed actual_final_T")
            if not complete and rendered != 4:
                raise ValidationError(f"Row {line}: completed-task rendered_T requires an economic terminal choice or four certified rendered revelations; an administrative partial rendered count is not exact")
        elif rendered_observed:
            raise ValidationError(f"Row {line}: rendered_depth_observed requires exact rendered_T")
        if math.isfinite(purchase) and purchase not in (0, 1):
            raise ValidationError(f"Row {line}: purchase must be 0, 1, or missing")
        if (math.isfinite(purchase) or math.isfinite(cs)) and not complete:
            raise ValidationError(f"Row {line}: purchase/induced_CS require a completed economic terminal choice")
        if math.isfinite(cs) and not INDUCED_CS_SUPPORT[0] <= cs <= INDUCED_CS_SUPPORT[1]:
            raise ValidationError(f"Row {line}: induced_CS must be within the registered support [-32,99]")
        expected_entry = (float(depth > 0) if math.isfinite(depth)
                          else (1.0 if opens > 0 else float("nan")))
        if math.isfinite(optional["entry"]):
            if not math.isfinite(expected_entry):
                raise ValidationError(f"Row {line}: entry requires a certified T or a strictly positive verified opening lower bound")
            if abs(optional["entry"]-expected_entry) > IDENTITY_TOLERANCE:
                raise ValidationError(f"Row {line}: entry disagrees with certified T or verified openings")
        if math.isfinite(optional["SC"]):
            if not math.isfinite(depth):
                raise ValidationError(f"Row {line}: target SC requires exact completed-task T; actual partial charges are not target expenditure")
            if abs(optional["SC"]-FEE_LEVELS[c]*depth) > IDENTITY_TOLERANCE:
                raise ValidationError(f"Row {line}: supplied SC violates its identity with completed-task T and fee")
        for name, (lower, upper) in OPTIONAL_CHOICE_SUPPORTS.items():
            value = optional[name]
            if math.isfinite(value):
                if not complete:
                    raise ValidationError(f"Row {line}: {name} requires a completed economic terminal choice")
                if not lower <= value <= upper:
                    raise ValidationError(f"Row {line}: {name} must be within [{lower},{upper}]")
        tc, rev, welfare = (optional[name] for name in ("TC", "REV", "W"))
        search_cost = FEE_LEVELS[c]*depth
        if math.isfinite(purchase) and purchase == 1 and depth == 0:
            raise ValidationError(f"Row {line}: purchase requires at least one price revelation")
        if math.isfinite(tc) and tc > 4*g+IDENTITY_TOLERANCE:
            raise ValidationError(f"Row {line}: TC exceeds current gamma times maximum distance")
        if purchase == 0:
            if any(math.isfinite(value) and abs(value) > IDENTITY_TOLERANCE for value in (tc, rev)):
                raise ValidationError(f"Row {line}: explicit nonpurchase has TC=REV=0")
            if math.isfinite(welfare) and math.isfinite(cs) and abs(welfare-cs) > IDENTITY_TOLERANCE:
                raise ValidationError(f"Row {line}: nonpurchase requires W=induced_CS")
            if math.isfinite(welfare) and math.isfinite(search_cost):
                outside = welfare+search_cost
                if not 49-IDENTITY_TOLERANCE <= outside <= 51+IDENTITY_TOLERANCE:
                    raise ValidationError(f"Row {line}: nonpurchase W+SC is outside the registered outside-option support")
        if purchase == 1:
            if math.isfinite(rev) and rev < 20-IDENTITY_TOLERANCE:
                raise ValidationError(f"Row {line}: purchased price REV is below the registered price support")
            if math.isfinite(tc) and g > 0:
                distance = tc/g
                if not 1 <= round(distance) <= 4 or abs(distance-round(distance)) > IDENTITY_TOLERANCE:
                    raise ValidationError(f"Row {line}: purchased TC/gamma must be seller distance 1,2,3,4")
            if all(math.isfinite(value) for value in (cs, rev, tc, search_cost)):
                if abs(cs-(120-rev-tc-search_cost)) > IDENTITY_TOLERANCE:
                    raise ValidationError(f"Row {line}: purchase induced_CS violates 120-REV-TC-SC")
            if all(math.isfinite(value) for value in (welfare, tc, search_cost)):
                if abs(welfare-(120-tc-search_cost)) > IDENTITY_TOLERANCE:
                    raise ValidationError(f"Row {line}: purchase W violates 120-TC-SC")
        if all(math.isfinite(value) for value in (welfare, cs, rev)):
            if abs(welfare-cs-rev) > IDENTITY_TOLERANCE:
                raise ValidationError(f"Row {line}: W must equal induced_CS+REV")
        signature = (pair, a)
        if person in participants and participants[person] != signature:
            raise ValidationError(f"Participant {person} changes pair or assigned ranking")
        participants[person] = signature
        pairs.setdefault(pair, {})[person] = a
        slot = (person, r)
        if slot in unique_slots:
            raise ValidationError(f"Duplicate participant-round slot: {slot}")
        unique_slots.add(slot)
        parsed.append((pair, person, a, r, c, g, depth, complete, opens, purchase, cs, depth_observed,
                       tc, rev, welfare, rendered, rendered_observed, actual_final))
    if len(pairs) != config.pairs:
        raise ValidationError(f"Expected {config.pairs} assigned pairs; found {len(pairs)}")
    if len(participants) != 2*config.pairs:
        raise ValidationError("Expected exactly two assigned participants per pair")
    for pair, members in pairs.items():
        if len(members) != 2 or sorted(members.values()) != [0, 1]:
            raise ValidationError(f"Pair {pair} does not have one participant per ranking")
    for person in participants:
        expected = {(person, r) for r in range(1, config.rounds+1)}
        if not expected.issubset(unique_slots):
            raise ValidationError(f"Participant {person} has omitted assigned round slots")
    if len(parsed) != 2*config.pairs*config.rounds:
        raise ValidationError("Input does not contain exactly all assigned slots")
    schedules = {}
    for pair, _, _, r, c, g, *_ in parsed:
        key = (pair, r)
        if key in schedules and schedules[key] != (c, g):
            raise ValidationError(f"Pair {pair}, round {r}: current assignments are not shared")
        schedules[key] = (c, g)
    parsed.sort(key=lambda z: (z[0], z[2], z[3]))
    pair_ids = sorted(pairs)
    pair_index = {pair: index for index, pair in enumerate(pair_ids)}
    return {
        "pair_id": [z[0] for z in parsed],
        "participant_id": [z[1] for z in parsed],
        "pair": np.asarray([pair_index[z[0]] for z in parsed], dtype=int),
        "rank": np.asarray([z[2] for z in parsed], dtype=int),
        "round": np.asarray([z[3]-1 for z in parsed], dtype=int),
        "cost": np.asarray([z[4] for z in parsed], dtype=int),
        "gamma": np.asarray([GAMMAS.index(z[5]) for z in parsed], dtype=int),
        "T": np.asarray([z[6] for z in parsed], dtype=float),
        "terminal_complete": np.asarray([z[7] for z in parsed], dtype=bool),
        "depth_observed": np.asarray([z[11] for z in parsed], dtype=bool),
        "observed_opens": np.asarray([z[8] for z in parsed], dtype=float),
        "purchase": np.asarray([z[9] for z in parsed], dtype=float),
        "induced_CS": np.asarray([z[10] for z in parsed], dtype=float),
        "TC": np.asarray([z[12] for z in parsed], dtype=float),
        "REV": np.asarray([z[13] for z in parsed], dtype=float),
        "W": np.asarray([z[14] for z in parsed], dtype=float),
        "optional_outcomes_provided": sorted(optional_outcomes_provided),
        "rendered_T": np.asarray([z[15] for z in parsed], dtype=float),
        "rendered_depth_observed": np.asarray([z[16] for z in parsed], dtype=bool),
        "rendered_columns_provided": rendered_columns_provided,
        "actual_final_T": np.asarray([z[17] for z in parsed], dtype=float),
        "actual_final_column_provided": actual_final_column_provided,
        "pair_labels": pair_ids,
    }


def load_csv(path, config):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError("CSV is empty")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValidationError("CSV has duplicate column names")
        return validate_rows(list(reader), config)


def cell_index(data, rounds):
    return (((data["rank"]*2+data["cost"])*3+data["gamma"])*rounds+data["round"])


def contrast(rounds, name, gamma=2):
    """Return a prespecified contrast on rank x cost x gamma x period means."""
    out = np.zeros(12*rounds)
    for a in range(2):
        for c in range(2):
            for g in range(3):
                s = int(g == gamma)-int(g == 0)
                if name == "interaction":
                    value = (2*a-1)*s/(2*rounds)
                elif name == "near_simple":
                    value = (a == 0)*s/(2*rounds)
                elif name == "far_simple":
                    value = (a == 1)*s/(2*rounds)
                elif name == "rank_at_gamma":
                    value = (2*a-1)*(g == gamma)/(2*rounds)
                elif name == "near_CS_positive_gamma":
                    value = (1-2*a)*(g > 0)/(4*rounds)
                elif name == "rank_pooled":
                    value = (2*a-1)/(6*rounds)
                elif name == "cost_pooled":
                    value = (2*c-1)/(6*rounds)
                elif name == "friction_pooled":
                    value = s/(4*rounds)
                elif name == "three_way":
                    value = (2*a-1)*(2*c-1)*s/rounds
                else:
                    raise ValueError(f"Unknown contrast {name}")
                start = ((a*2+c)*3+g)*rounds
                out[start:start+rounds] = value
    return out


def restrict_cost(weights, rounds, cost):
    """Convert an equal-cost average to its cost-specific contrast."""
    if cost not in (0, 1):
        raise ValueError("Cost code must be 0 or 1")
    cube = np.asarray(weights).reshape(2, 2, 3, rounds).copy()
    cube[:, 1-cost] = 0
    cube[:, cost] *= 2
    return cube.reshape(-1)


def restrict_periods(weights, rounds, start, stop):
    """Equal average within [start,stop), retaining the full period design."""
    if not 0 <= start < stop <= rounds:
        raise ValueError("Invalid period range")
    cube = np.asarray(weights).reshape(12, rounds).copy()
    cube[:, :start] = 0
    cube[:, stop:] = 0
    cube *= rounds/(stop-start)
    return cube.reshape(-1)


def derive_depth_outcomes(data):
    """Target T identifies SC; any verified opening also identifies positive entry."""
    result = dict(data)
    depth = np.asarray(data["T"], dtype=float)
    observed = np.isfinite(depth)
    opens = np.asarray(data.get("observed_opens", np.zeros_like(depth)), dtype=float)
    result["entry"] = np.where(observed, (depth > 0).astype(float), np.where(opens > 0, 1.0, np.nan))
    result["SC"] = np.where(observed, np.asarray(FEE_LEVELS)[data["cost"]]*depth, np.nan)
    return result


def cr2(data, outcome, weights, config, geometry=False):
    """Saturated observed means + pair CR2, identity-working contrast df."""
    x = cell_index(data, config.rounds)
    y = np.asarray(data[outcome], dtype=float)
    keep = np.isfinite(y)
    xo, yo, bo = x[keep], y[keep], data["pair"][keep]
    n_cells = 12*config.rounds
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_cells,) or not np.any(weights):
        raise EstimationError("Contrast weights must be nonzero and have 12*J entries")
    n = np.bincount(xo, minlength=n_cells)
    needed = weights != 0
    if np.any(n[needed] == 0):
        raise EstimationError("A contrast-relevant observed cell is empty")
    if np.any(n[needed] == 1):
        raise EstimationError("A contrast-relevant cell is singleton; CR2 undefined")
    occurrence = np.bincount(bo*n_cells+xo, minlength=config.pairs*n_cells)
    if np.any(occurrence > 1):
        raise EstimationError("CR2 shortcut requires at most one record per pair per full cell")
    sums = np.bincount(xo, weights=yo, minlength=n_cells)
    mu = np.divide(sums, n, out=np.zeros(n_cells), where=n > 0)
    residual = yo-mu[xo]
    ell = np.zeros(n_cells)
    ell[needed] = weights[needed]/(n[needed]*np.sqrt(1-1/n[needed]))
    score = np.bincount(bo, weights=ell[xo]*residual, minlength=config.pairs)
    variance = float(np.dot(score, score))
    diagonal = np.bincount(bo, weights=ell[xo]**2, minlength=config.pairs)
    v = np.zeros((config.pairs, n_cells))
    v[bo, xo] = ell[xo]
    inv_n = np.divide(1.0, n, out=np.zeros(n_cells), where=n > 0)
    G = np.diag(diagonal)-np.einsum("bx,cx,x->bc", v, v, inv_n, optimize=False)
    trace = float(np.trace(G))
    denominator = float(np.einsum("bc,bc->", G, G))
    if trace <= 0 or denominator <= 0:
        raise EstimationError("Degenerate contrast information geometry")
    df = trace**2/denominator
    target_trace = float(np.sum(weights[needed]**2/n[needed]))
    if not np.isclose(trace, target_trace, rtol=1e-10, atol=1e-12):
        raise AssertionError("Identity-working-model CR2 trace check failed")
    if df > config.pairs-1+1e-8:
        raise AssertionError("Contrast df exceeds independent-pair rank bound")
    estimate = float(np.dot(weights, mu))
    se = math.sqrt(variance)
    outcome_scale = max(1.0, float(np.max(np.abs(yo))))
    numerical_se_tolerance = NUMERICAL_SE_REL_TOLERANCE*outcome_scale
    result = {
        "status": "ok", "outcome": outcome, "estimate": estimate, "se": se,
        "variance": variance, "df": df, "assigned_pairs": config.pairs,
        "min_relevant_cell_n": int(n[needed].min()),
        "max_relevant_cell_n": int(n[needed].max()),
        "observed_records": int(keep.sum()),
        "outcome_scale": outcome_scale,
        "numerical_se_tolerance": numerical_se_tolerance,
    }
    if se <= numerical_se_tolerance:
        result.update(status="zero_variance_no_automatic_test", p_greater=None,
                      p_less=None, p_two_sided=None, ci95=None,
                      degeneracy_reason="SE does not exceed the fixed scale-aware numerical tolerance")
    else:
        statistic = estimate/se
        critical = student_t.ppf(0.975, df)
        result.update(t=statistic, p_greater=float(student_t.sf(statistic, df)),
                      p_less=float(student_t.cdf(statistic, df)),
                      p_two_sided=float(2*student_t.sf(abs(statistic), df)),
                      ci95=[estimate-critical*se, estimate+critical*se])
    if geometry:
        result["G"] = G
        result["pair_scores"] = score
        result["full_cell_means"] = mu
    return result


def safe_cr2(data, outcome, weights, config):
    try:
        return cr2(data, outcome, weights, config)
    except EstimationError as exc:
        return {"status": "not_estimable", "outcome": outcome, "reason": str(exc)}


def tost(result, margin, alpha=0.05):
    """TOST of a difference being strictly within the proposed +/- margin."""
    tolerance = result.get("numerical_se_tolerance", NUMERICAL_SE_REL_TOLERANCE
                           *max(1.0, result.get("outcome_scale", 1.0)))
    if result.get("status") != "ok" or result.get("se", 0) <= tolerance:
        return {"status": "not_estimable", "margin": margin, "p": None,
                "reason": "Underlying contrast has no valid standard error"}
    estimate, se, df = result["estimate"], result["se"], result["df"]
    lower_p = float(student_t.sf((estimate+margin)/se, df))
    upper_p = float(student_t.cdf((estimate-margin)/se, df))
    p = max(lower_p, upper_p)
    critical = student_t.ppf(1-alpha, df)
    return {"status": "ok", "margin": margin, "p": p,
            "p_lower": lower_p, "p_upper": upper_p,
            "ci_1_minus_2alpha": [estimate-critical*se, estimate+critical*se],
            "equivalent_unadjusted": p < alpha}


def joint_max(pvalues):
    if any(p is None for p in pvalues):
        return None
    return max(pvalues)


def joint_signs_primary(near, far, alpha=0.05):
    """Primary IUT: near endpoint effect <0 AND far endpoint effect >0."""
    components = {}
    for label, result, field in (("near_negative", near, "p_less"),
                                 ("far_positive", far, "p_greater")):
        p = result.get(field)
        available = (result.get("status") == "ok" and isinstance(p, (int, float))
                     and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1)
        components[label] = {"raw_p": float(p) if available else None,
                             "effective_p": float(p) if available else 1.0,
                             "status": "ok" if available else "not_estimable",
                             "contrast_status": result.get("status", "not_estimable")}
    unavailable = [name for name, value in components.items() if value["status"] != "ok"]
    p = max(value["effective_p"] for value in components.values())
    return {"status": "not_estimable" if unavailable else "ok",
            "registered_null": "Delta_near >= 0 OR Delta_far <= 0",
            "registered_alternative": "Delta_near < 0 AND Delta_far > 0",
            "primary_p": p, "p": p,
            "reject_primary": bool(not unavailable and p < alpha),
            "component_pvalues": components, "unavailable_components": unavailable,
            "rule": "maximum of the negative-near and positive-far one-sided p-values",
            "decision_uses_Holm_family": False,
            "classification": "primary_joint_directional_signs"}


def holm(pvalues, alpha=0.05):
    """Fixed-family Holm adjustment; unavailable tests count as p=1."""
    names = list(pvalues)
    effective = np.asarray([1.0 if pvalues[n] is None else pvalues[n] for n in names])
    order = np.argsort(effective, kind="stable")
    adjusted = np.zeros(len(names))
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(names)-position)*effective[index])
        adjusted[index] = min(1.0, running)
    return {name: {"raw_p": pvalues[name], "holm_p": float(adjusted[i]),
                   "reject": bool(pvalues[name] is not None and adjusted[i] < alpha),
                   "status": "not_estimable" if pvalues[name] is None else "ok"}
            for i, name in enumerate(names)}


def describe_cells(data, outcome, config):
    x = cell_index(data, config.rounds)
    y = data[outcome]
    records = []
    for a in range(2):
        for c in range(2):
            for g in range(3):
                period_means = []
                for r in range(config.rounds):
                    key = ((a*2+c)*3+g)*config.rounds+r
                    select = x == key
                    observed = y[select & np.isfinite(y)]
                    mean = float(observed.mean()) if len(observed) else None
                    period_means.append(mean)
                    records.append({"rank": a, "cost": c, "gamma": GAMMAS[g],
                                    "round": r+1, "assigned_n": int(select.sum()),
                                    "observed_n": len(observed), "mean": mean})
                # Do not silently reweight periods when one is unobserved.
                records.append({"rank": a, "cost": c, "gamma": GAMMAS[g],
                                "round": "equally_standardised_over_periods",
                                "mean": None if None in period_means else float(np.mean(period_means))})
    return records


def ht_bounds(data, config, contrast_name="interaction"):
    """Exact completion extrema of the HT statistic; pair-level uncertainty."""
    config.validate()
    p0, _, p4 = config.gamma_probabilities
    gamma_sign = (data["gamma"] == 2).astype(float)/p4-(data["gamma"] == 0)/p0
    if contrast_name == "interaction":
        row_weight = 1/config.rounds*(2*data["rank"]-1)*gamma_sign
    elif contrast_name == "near_simple":
        row_weight = 1/config.rounds*(data["rank"] == 0)*gamma_sign
    elif contrast_name == "far_simple":
        row_weight = 1/config.rounds*(data["rank"] == 1)*gamma_sign
    else:
        raise ValueError("Bounds support interaction and the two simple effects")
    observed = np.isfinite(data["T"])
    low_y = np.where(observed, data["T"], data["observed_opens"])
    high_y = np.where(observed, data["T"], 4.0)
    lower_row = row_weight*np.where(row_weight >= 0, low_y, high_y)
    upper_row = row_weight*np.where(row_weight >= 0, high_y, low_y)
    lower_scores = np.bincount(data["pair"], weights=lower_row, minlength=config.pairs)
    upper_scores = np.bincount(data["pair"], weights=upper_row, minlength=config.pairs)
    lo, hi = float(lower_scores.mean()), float(upper_scores.mean())
    se_lo = float(lower_scores.std(ddof=1)/math.sqrt(config.pairs))
    se_hi = float(upper_scores.std(ddof=1)/math.sqrt(config.pairs))
    # Two one-sided alpha/2 bounds: Bonferroni envelope. Student reference is
    # cluster-asymptotic, not a finite-exact or guaranteed conservative t law.
    critical = float(student_t.ppf(1-config.alpha/2, config.pairs-1))
    # Every endpoint pair score lies in +/-4/min(p0,p4). The same valid (possibly
    # loose) range covers interaction and simple effects, including missing T.
    score_bound = 4/min(p0, p4)
    radius = 2*score_bound*math.sqrt(math.log(2/config.alpha)/(2*config.pairs))
    return {
        "contrast": contrast_name, "missing_T_slots": int((~observed).sum()),
        "classification": "supporting_partial_identification_not_primary_point_test",
        "gamma_probabilities": list(config.gamma_probabilities),
        "pair_endpoint_score_support": [-score_bound, score_bound],
        "assigned_slots": len(observed),
        "exact_HT_completion_extrema": [lo, hi],
        "lower_endpoint_se": se_lo, "upper_endpoint_se": se_hi,
        "pair_t_Bonferroni_envelope": [lo-critical*se_lo, hi+critical*se_hi],
        "pair_t_reference_df": config.pairs-1,
        "pair_t_envelope_status": "approximate; Bonferroni over two endpoint tails",
        "finite_Hoeffding_envelope": [lo-radius, hi+radius],
        "Hoeffding_status": "finite-sample conservative under independent bounded pair scores",
        "positive_interaction_pair_t_support_gate":
            bool(contrast_name == "interaction" and lo-critical*se_lo > 0),
        "positive_interaction_finite_Hoeffding_gate":
            bool(contrast_name == "interaction" and lo-radius > 0),
        "no_MAR_assumption": True,
        "positive_interaction_alone_is_not_primary_reversal_support": True,
    }


def joint_signs_bound_support(bounds):
    """Conservative IUT gates from existing separate two-sided envelopes."""
    near, far = bounds["near_simple"], bounds["far_simple"]
    return {
        "finite_Hoeffding_IUT_gate": bool(near["finite_Hoeffding_envelope"][1] < 0
                                           and far["finite_Hoeffding_envelope"][0] > 0),
        "approximate_pair_t_IUT_gate": bool(near["pair_t_Bonferroni_envelope"][1] < 0
                                             and far["pair_t_Bonferroni_envelope"][0] > 0),
        "rule": "near upper envelope < 0 AND far lower envelope > 0",
        "classification": "conservative_IUT_support_gate_not_a_joint_confidence_interval",
        "component_directional_tail": "alpha/2 from each existing two-sided 1-alpha envelope",
        "finite_gate_validity": "independent bounded pair scores and registered assignment probabilities",
        "pair_t_gate_validity": "approximate cluster inference; not a finite-sample guarantee",
        "no_MAR_assumption": True,
        "positive_interaction_gate_suffices": False,
    }


def _ri_components(data, config):
    """Pair/cell sufficient quantities for exactly the raw standardised statistic."""
    base_cell = (data["cost"]*3+data["gamma"])*config.rounds+data["round"]
    n_base = 6*config.rounds
    values = np.zeros((2, config.pairs, n_base))
    observed = np.zeros_like(values)
    good = np.isfinite(data["T"])
    idx = (data["rank"][good], data["pair"][good], base_cell[good])
    values[idx] = data["T"][good]
    observed[idx] = 1
    base_weights = np.zeros(n_base)
    for c in range(2):
        base_weights[(c*3)*config.rounds:(c*3+1)*config.rounds] = -1/(2*config.rounds)
        base_weights[(c*3+2)*config.rounds:(c*3+3)*config.rounds] = 1/(2*config.rounds)
    needed = base_weights != 0
    # One fully observed pair in every relevant C/G/period cell ensures the
    # point statistic is defined under EVERY ranking-coin allocation, not just
    # the sampled allocations. Do not discard undefined permutations.
    if np.any(np.sum(observed[0]*observed[1], axis=0)[needed] == 0):
        raise EstimationError("RI needs a fully observed pair in every relevant cost/gamma/period cell so all flips have defined raw means")
    return (0.5*(values[1]+values[0]).sum(axis=0)[needed],
            0.5*(values[1]-values[0])[:, needed],
            0.5*(observed[1]+observed[0]).sum(axis=0)[needed],
            0.5*(observed[1]-observed[0])[:, needed], base_weights[needed])


def _ri_statistics(signs, components):
    values_mid, values_delta, count_mid, count_delta, weights = components
    dv = np.einsum("db,bk->dk", signs, values_delta, optimize=False)
    dn = np.einsum("db,bk->dk", signs, count_delta, optimize=False)
    far = (values_mid+dv)/(count_mid+dn)
    near = (values_mid-dv)/(count_mid-dn)
    return np.einsum("dk,k->d", far-near, weights, optimize=False)


def sharp_path_ri(data, config):
    """Raw-cell point-statistic RI under a full-path plus observation sharp null."""
    try:
        components = _ri_components(data, config)
    except EstimationError as exc:
        return {"status": "not_estimable", "reason": str(exc), "p_plus_one": None,
                "classification": "supporting_interaction_sharp_null_check_only",
                "not_a_test_of_primary_joint_sign_null": True}
    observed_statistic = float(_ri_statistics(np.ones((1, config.pairs)), components)[0])
    rng = np.random.default_rng(config.ri_seed)
    count = 0
    draws_remaining = config.ri_draws
    while draws_remaining:
        batch = min(draws_remaining, 1000)
        signs = rng.integers(0, 2, size=(batch, config.pairs))*2-1
        permuted = _ri_statistics(signs, components)
        tolerance = 1e-12*max(1.0, abs(observed_statistic))
        count += int(np.sum(permuted >= observed_statistic-tolerance))
        draws_remaining -= batch
    return {
        "status": "ok",
        "statistic": "raw full-cell standardised endpoint interaction; supporting S1 statistic, not the primary joint-sign IUT",
        "classification": "supporting_interaction_sharp_null_check_only",
        "not_a_test_of_primary_joint_sign_null": True,
        "studentised": False,
        "observed_statistic": observed_statistic, "alternative": "greater",
        "draws": config.ri_draws, "seed": config.ri_seed,
        "p_plus_one": (count+1)/(config.ri_draws+1),
        "sharp_null": "ranking leaves each participant's entire depth path AND observation pattern unchanged conditional on shared schedules and raw profiles",
        "conditioning": "fixed pair membership, participants, current cost/friction schedules and raw profiles; only within-pair ranking coins are redrawn",
        "not_a_test_of_weak_scientific_interaction_null": True,
        "missing_T_imputed": False,
        "Monte_Carlo_status": "plus-one randomisation p-value; not exhaustive enumeration",
    }


def analyse(data, config):
    data = derive_depth_outcomes(data)
    J = config.rounds
    effect = lambda outcome, name, g=2: safe_cr2(data, outcome, contrast(J, name, g), config)
    config.validate()
    interaction = effect("T", "interaction")
    interaction["registered_alternative"] = "theta > 0"
    interaction["classification"] = "secondary_S1_positive_endpoint_interaction"
    interaction["decision_uses_Holm_family"] = True
    near, far = effect("T", "near_simple"), effect("T", "far_simple")
    primary = joint_signs_primary(near, far, config.alpha)
    zero_depth = effect("T", "rank_at_gamma", 0)
    depth_tost = tost(zero_depth, config.depth_equivalence_margin, config.alpha)
    conversion = {}
    for g in range(3):
        raw = effect("purchase", "rank_at_gamma", g)
        conversion[str(GAMMAS[g])] = {
            "contrast": raw,
            "TOST": tost(raw, config.conversion_equivalence_margin, config.alpha),
        }
    conversion_joint_p = joint_max([z["TOST"]["p"] for z in conversion.values()])
    cs = effect("induced_CS", "near_CS_positive_gamma")
    family = holm({"positive_endpoint_interaction": interaction.get("p_greater"),
                   "gamma0_depth_ranking_equivalence": depth_tost["p"],
                   "conversion_ranking_equivalence_all_gamma": conversion_joint_p,
                   "near_CS_advantage_positive_gamma": cs.get("p_greater")}, config.alpha)
    diagnostic = {"ranking_pooled": effect("T", "rank_pooled"),
                  "cost_pooled": effect("T", "cost_pooled"),
                  "friction_2_vs_0_pooled": effect("T", "friction_pooled", 1),
                  "friction_4_vs_0_pooled": effect("T", "friction_pooled", 2),
                  "interaction_2_vs_0": effect("T", "interaction", 1),
                  "three_way_endpoint": effect("T", "three_way", 2)}
    depth_exploratory = {
        "status": "exploratory_only_not_additional_confirmatory_discoveries",
        "endpoint_effects_by_fee": {},
        "ranking_difference_at_gamma": {str(GAMMAS[g]): effect("T", "rank_at_gamma", g)
                                         for g in range(3)},
        "friction_increments": {},
        "high_minus_low_fee": diagnostic["cost_pooled"],
        "fee_difference_in_endpoint_interaction": diagnostic["three_way_endpoint"],
    }
    for c in range(2):
        depth_exploratory["endpoint_effects_by_fee"][str(int(FEE_LEVELS[c]))] = {
            name: safe_cr2(data, "T", restrict_cost(contrast(J, name), J, c), config)
            for name in ("near_simple", "far_simple", "interaction")}
    for low, high in ((0, 1), (1, 2)):
        increment = {}
        for name in ("near_simple", "far_simple", "interaction", "friction_pooled"):
            weights = contrast(J, name, high)-contrast(J, name, low)
            increment[name] = safe_cr2(data, "T", weights, config)
        depth_exploratory["friction_increments"][f"{GAMMAS[low]}_to_{GAMMAS[high]}"] = increment
    if J % 2 == 0:
        half = J//2
        first = restrict_periods(contrast(J, "interaction"), J, 0, half)
        last = restrict_periods(contrast(J, "interaction"), J, half, J)
        depth_exploratory["period_halves"] = {
            "first_rounds": [1, half], "last_rounds": [half+1, J],
            "first_half": safe_cr2(data, "T", first, config),
            "last_half": safe_cr2(data, "T", last, config),
            "last_minus_first": safe_cr2(data, "T", last-first, config),
        }
    else:
        depth_exploratory["period_halves"] = {
            "status": "not_applicable", "reason": "Prespecified equal halves require an even J"}
    optional_names = data.get("optional_outcomes_provided",
                              [name for name in OPTIONAL_CHOICE_SUPPORTS if name in data])
    exploratory_names = ["entry", "SC"]+[name for name in OPTIONAL_CHOICE_SUPPORTS if name in optional_names]
    exploratory_outcomes = {}
    for outcome in exploratory_names:
        exploratory_outcomes[outcome] = {
            "status": "exploratory_only_not_additional_confirmatory_discoveries",
            "ranking_pooled": effect(outcome, "rank_pooled"),
            "high_minus_low_fee": effect(outcome, "cost_pooled"),
            "endpoint_interaction": effect(outcome, "interaction"),
            "friction_0_to_2_pooled": effect(outcome, "friction_pooled", 1),
            "friction_2_to_4_pooled": safe_cr2(data, outcome,
                contrast(J, "friction_pooled", 2)-contrast(J, "friction_pooled", 1), config),
        }
    rendered_provided = data.get("rendered_columns_provided", "rendered_T" in data)
    rendered_count = int(np.isfinite(data["rendered_T"]).sum()) if "rendered_T" in data else 0
    if not rendered_provided or rendered_count == 0:
        rendered_interaction = {
            "status": "not_available", "observed_records": rendered_count,
            "reason": "No certified exact rendered-depth outcomes were supplied; committed T is not substituted",
        }
        rendered_near, rendered_far = dict(rendered_interaction), dict(rendered_interaction)
    else:
        rendered_interaction = safe_cr2(data, "rendered_T", contrast(J, "interaction"), config)
        rendered_near = safe_cr2(data, "rendered_T", contrast(J, "near_simple"), config)
        rendered_far = safe_cr2(data, "rendered_T", contrast(J, "far_simple"), config)
    rendered_primary = joint_signs_primary(rendered_near, rendered_far, config.alpha)
    rendered_primary.update(outcome="rendered_T",
                            classification="rendered_depth_joint_signs_sensitivity_not_confirmatory",
                            may_replace_committed_primary=False)
    # Keep the previous top-level interaction fields as supporting compatibility
    # outputs; the explicit nested records distinguish all rendered calculations.
    rendered_sensitivity = dict(rendered_interaction)
    rendered_sensitivity.update(
        classification="sensitivity_only_not_an_additional_confirmatory_claim",
        alternative="Delta_near_rendered < 0 AND Delta_far_rendered > 0; theta_rendered supporting",
        near_simple=rendered_near, far_simple=rendered_far,
        primary_joint_signs=rendered_primary, interaction_supporting=rendered_interaction,
        may_replace_committed_primary=False,
        p_value_selection_rule="report all prespecified sensitivities; never choose the most favorable p-value",
        observation_assumption="rendered-outcome-specific mean observation exchangeability within rank/cost/gamma/period",
        certification="reconciled persistent render acknowledgements plus completed economic terminal choice, or four certified rendered revelations",
    )
    actual_final_provided = data.get("actual_final_column_provided", "actual_final_T" in data)
    actual_final = np.asarray(data.get("actual_final_T", np.full_like(data["T"], np.nan)), dtype=float)
    actual_final_known = np.isfinite(actual_final)
    bounds = {name: ht_bounds(data, config, name)
              for name in ("interaction", "near_simple", "far_simple")}
    return {
        "analysis_version": ANALYSIS_VERSION,
        "config": asdict(config),
        "fixed_analysis_rules": {
            "numerical_se_relative_tolerance": NUMERICAL_SE_REL_TOLERANCE,
            "induced_CS_support": list(INDUCED_CS_SUPPORT),
            "choice_outcomes_require_terminal_complete": True,
            "target_depth_certification": "economic terminal choice and exact count, or verified observed_opens=T=4",
            "administrative_partial_depth": "descriptive actual_final_T only; target T missing with observed_opens lower bound",
            "entry_certification": "certified T, or positive verified observed_opens for entry=1; never actual_final_T alone",
            "fee_levels": list(FEE_LEVELS),
            "optional_choice_supports": OPTIONAL_CHOICE_SUPPORTS,
            "outcome_identity_tolerance": IDENTITY_TOLERANCE,
            "primary_claim": "negative near endpoint effect AND positive far endpoint effect",
            "secondary_claim_order": ["positive_endpoint_interaction", "gamma0_depth_ranking_equivalence",
                                      "conversion_ranking_equivalence_all_gamma", "near_CS_advantage_positive_gamma"],
        },
        "interpretation": {
            "point_analysis": "conditional on outcome-specific current full-cell mean observation exchangeability",
            "target": "completed-task outcome under the registered recursively completed presentation regime; administrative partial counts are not completed-task outcomes",
            "clustering": "independent assigned matched pairs, including incomplete pairs",
            "inference": "CR2 identity working covariance and contrast Satterthwaite; approximate not finite exact",
            "margins": "0.25 depth and 0.05 conversion are proposed substantive margins, not theory-derived",
            "diagnostics": "unadjusted descriptive/exploratory contrasts, not additional registered discoveries",
        },
        "accounting": {
            "assigned_pairs": config.pairs, "assigned_participants": 2*config.pairs,
            "assigned_slots": 2*config.pairs*J,
            "observed_T": int(np.isfinite(data["T"]).sum()),
            "zero_T_retained": int(np.sum(data["T"] == 0)),
            "observed_T_without_complete_economic_choice":
                int(np.sum(np.isfinite(data["T"]) & ~data["terminal_complete"])),
            "observed_purchase": int(np.isfinite(data["purchase"]).sum()),
            "observed_induced_CS": int(np.isfinite(data["induced_CS"]).sum()),
            "observed_exploratory_outcomes": {name: int(np.isfinite(data[name]).sum())
                                              for name in exploratory_names},
            "optional_choice_outcomes_provided": list(optional_names),
            "certified_rendered_depth_records": rendered_count,
            "descriptive_actual_final_depth_records": int(actual_final_known.sum()),
        },
        "primary_joint_signs": primary,
        "interaction_supporting": interaction,
        "near_simple": near, "far_simple": far,
        "directional_sign_reversal_IUT": dict(primary),
        "gamma0_depth_rank_contrast": zero_depth,
        "gamma0_depth_rank_TOST": depth_tost,
        "conversion_by_gamma": conversion,
        "conversion_joint_TOST": {"p": conversion_joint_p, "rule": "maximum of three gamma-specific TOST p-values"},
        "near_CS_advantage_positive_gamma": cs,
        "secondary_Holm_family": family,
        "diagnostics": diagnostic,
        "depth_exploratory": depth_exploratory,
        "exploratory_outcomes": exploratory_outcomes,
        "rendered_revelation_sensitivity": rendered_sensitivity,
        "actual_final_depth_descriptive": {
            "status": "descriptive_only_not_the_completed_task_target" if actual_final_provided else "not_provided",
            "observed_records": int(actual_final_known.sum()),
            "observed_mean": float(actual_final[actual_final_known].mean()) if actual_final_known.any() else None,
            "actual_depth_without_economic_choice": int(np.sum(actual_final_known & ~data["terminal_complete"])),
            "cell_summaries": describe_cells(data, "actual_final_T", config) if actual_final_provided else [],
            "used_to_impute_target_outcomes": False,
        },
        "cell_means": {outcome: describe_cells(data, outcome, config)
                       for outcome in ["T", "purchase", "induced_CS"]+exploratory_names
                       +(["rendered_T"] if rendered_provided else [])},
        "HT_depth_bounds": bounds,
        "primary_joint_signs_bound_support": joint_signs_bound_support(bounds),
        "supplementary_sharp_path_RI": sharp_path_ri(data, config),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV containing every originally assigned slot")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--pairs", type=int, default=150)
    parser.add_argument("--ri-draws", type=int, default=9999)
    parser.add_argument("--ri-seed", type=int, default=2026091324)
    parser.add_argument("--gamma-probabilities", type=float, nargs=3, default=(0.4, 0.2, 0.4),
                        metavar=("P0", "P2", "P4"), help="Frozen allocation default: 0.4 0.2 0.4")
    parser.add_argument("--output", type=Path, help="New JSON output path; existing files are never overwritten")
    args = parser.parse_args()
    config = Config(rounds=args.rounds, pairs=args.pairs, ri_draws=args.ri_draws, ri_seed=args.ri_seed,
                    gamma_probabilities=tuple(args.gamma_probabilities))
    try:
        data = load_csv(args.csv, config)
        report = analyse(data, config)
    except (ValidationError, OSError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        # Explicit user-named export, exclusive creation: never overwrite an analysis.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded+"\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
