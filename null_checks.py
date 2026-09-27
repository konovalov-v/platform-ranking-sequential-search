"""Pre-data size checks on both branches of the primary union null."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "power_helpers.py"
spec = importlib.util.spec_from_file_location("joint_null_batch_reference", SOURCE)
batch_reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch_reference
spec.loader.exec_module(batch_reference)

CASES = {
    "near_boundary_far_positive": {"beta": (0., .12), "loss": (0., 0.)},
    "near_negative_far_boundary": {"beta": (-.12, 0.), "loss": (0., 0.)},
    "near_boundary_differential_person_loss": {"beta": (0., .12), "loss": (.1, .3)},
    "near_wrong_sign_far_positive": {"beta": (.04, .12), "loss": (.1, .1)},
}
PROBABILITIES = (.4, .2, .4)


def generate(rng, count, specification):
    """Identity-link Binomial(4,p) marginals with persistent clustered noise."""
    p, j = batch_reference.P, batch_reference.J
    c = rng.integers(0, 2, (count, p, j))
    g = rng.choice(3, (count, p, j), p=PROBABILITIES)
    x = g / 2.
    lag_g = np.concatenate((np.zeros((count, p, 1)), x[:, :, :-1]-.5), axis=2)
    lag_c = np.concatenate((np.zeros((count, p, 1)), c[:, :, :-1]-.5), axis=2)
    cumulative = np.zeros_like(x)
    cumulative[:, :, 1:] = np.cumsum(x-.5, axis=2)[:, :, :-1] / np.arange(1, j)
    sign = lambda shape: 2*rng.integers(0, 2, shape)-1
    shared = .10*sign((count, p, 1, 1))
    individual = .04*sign((count, p, 2, 1))
    slope = .05*sign((count, p, 2, 1))
    beta = np.asarray(specification["beta"])[None, None, :, None]
    probabilities = (.5+shared+individual-.03*c[:, :, None]
                     +(beta+slope)*x[:, :, None]+.08*lag_g[:, :, None]
                     +.04*lag_c[:, :, None]+.04*cumulative[:, :, None])
    assert np.all((probabilities > 0) & (probabilities < 1))
    # 80% of rounds reuse the person's four latent uniforms, preserving each
    # round's Binomial marginal but making within-person dependence strong.
    permanent = rng.random((count, p, 2, 1, 4))
    fresh = rng.random((count, p, 2, j, 4))
    persistent = rng.random((count, p, 2, j, 1)) < .8
    depth = (np.where(persistent, permanent, fresh) < probabilities[..., None]).sum(axis=-1)
    observed = rng.random((count, p, 2)) >= np.asarray(specification["loss"])[None, None]
    outcomes = np.stack((depth, np.zeros_like(depth), depth)).astype(float)
    return (c, g), outcomes, observed, [float(probabilities.min()), float(probabilities.max())]


def analyse(design, outcomes, observed):
    geometry = batch_reference.geometry(*design, observed)
    result = batch_reference.analyse_batch(outcomes, geometry)
    estimates, se, df = (result[k][:, 1:3] for k in ("estimate", "se", "df"))
    valid = result["valid"][:, 1:3]
    statistics = np.divide(estimates, se, out=np.zeros_like(estimates), where=se > 0)
    p_near = np.where(valid[:, 0], student_t.cdf(statistics[:, 0], df[:, 0]), 1.)
    p_far = np.where(valid[:, 1], student_t.sf(statistics[:, 1], df[:, 1]), 1.)
    result["joint_primary_p"] = np.maximum(p_near, p_far)
    result["component_p"] = np.column_stack((p_near, p_far))
    return result


def verify_reference(design, outcomes, observed, result):
    errors = []
    for k in range(min(2, len(observed))):
        data = batch_reference.reference_data(design, outcomes, observed, k)
        components = []
        for column, name in ((1, "near_simple"), (2, "far_simple")):
            reference = batch_reference.reference.cr2(
                data, "T", batch_reference.reference.contrast(batch_reference.J, name),
                batch_reference.reference.Config())
            for field in ("estimate", "se", "df"):
                error = abs(reference[field]-result[field][k, column])
                errors.append(error)
                assert np.isclose(reference[field], result[field][k, column], rtol=1e-10, atol=1e-10)
            components.append(reference)
        primary = batch_reference.reference.joint_signs_primary(*components)
        assert np.isclose(primary["primary_p"], result["joint_primary_p"][k], atol=1e-12)
    return errors


def rate(values):
    values = np.asarray(values, dtype=bool)
    q = float(values.mean())
    return {"rate": q, "successes": int(values.sum()), "replications": len(values),
            "Monte_Carlo_SE": float(np.sqrt(q*(1-q)/len(values)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1710)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/null")
    args = parser.parse_args()
    if args.replications < 2 or args.batch < 1:
        parser.error("At least two replications and positive batch size required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {"status": "SYNTHETIC WEAK-UNION-NULL STRESS, NOT SUBJECT DATA",
              "analysis_version": batch_reference.reference.ANALYSIS_VERSION,
              "participants": 300, "pairs": 150, "rounds": 24,
              "gamma_probabilities": PROBABILITIES, "replications_per_case": args.replications,
              "batch_size": args.batch, "seed": args.seed,
              "conditional_DGP": "T|probability primitives~Binomial(4,p); unconditionally a mixture; identity-link p, shared/person intercepts, mean-zero slopes and past histories",
              "saved_effect_names": ["supporting_depth_interaction", "near_endpoint_depth", "far_endpoint_depth"],
              "auxiliary_outcomes_saved_or_interpreted": False,
              "dependence": "80% reuse of four person-specific independent latent uniforms; shared pair intercept",
              "point_observation": "rank-specific person loss independent of every latent outcome and task primitive",
              "null_scope": "Two weak-union boundary branches, one differential-loss boundary, and one wrong-sign interior case",
              "limitations": "Approximate CR2/IUT size in these cases only; not uniform finite-sample validity. Observation-dependent loss is not covered.",
              "sources_sha256": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), SOURCE, HERE / "experiment_analysis.py")},
              "cases": {}}
    arrays = {}
    for index, (name, specification) in enumerate(CASES.items()):
        seed = args.seed+index
        rng = np.random.default_rng(seed)
        buffers, errors = [], []
        probability_support = [1., 0.]
        for offset in range(0, args.replications, args.batch):
            count = min(args.batch, args.replications-offset)
            design, outcomes, observed, support = generate(rng, count, specification)
            result = analyse(design, outcomes, observed)
            if offset == 0:
                errors.extend(verify_reference(design, outcomes, observed, result))
            for k, reducer in enumerate((min, max)):
                probability_support[k] = reducer(probability_support[k], support[k])
            # The batching API also computes dummy secondary outcomes. They are
            # neither saved nor interpreted as purchase/payoff stress evidence.
            buffers.append({k: (result[k][:, :3] if k in ("estimate", "se", "df", "valid") else result[k])
                            for k in ("estimate", "se", "df", "valid", "joint_primary_p", "component_p")})
            if offset % (args.batch*20) == 0:
                print(json.dumps({"case": name, "completed": offset+count, "elapsed_seconds": time.monotonic()-start}), flush=True)
        all_results = {k: np.concatenate([x[k] for x in buffers]) for k in buffers[0]}
        means = 4*np.asarray(specification["beta"])
        estimates, se, df = (all_results[k][:, 1:3] for k in ("estimate", "se", "df"))
        valid = all_results["valid"][:, 1:3]
        coverage = valid & (np.abs(estimates-means) <= student_t.ppf(.975, df)*se)
        report["cases"][name] = {
            "seed": seed, "population_near_far_endpoint_effects": means.tolist(),
            "population_interaction": float(means[1]-means[0]),
            "person_loss_by_rank": specification["loss"],
            "joint_primary_rejection": rate(all_results["joint_primary_p"] < .05),
            "near_component_rejection": rate(all_results["component_p"][:, 0] < .05),
            "far_component_rejection": rate(all_results["component_p"][:, 1] < .05),
            "near_95CI_coverage": rate(coverage[:, 0]), "far_95CI_coverage": rate(coverage[:, 1]),
            "component_nonestimable_counts": (~valid).sum(axis=0).tolist(),
            "empirical_mean_near_far": estimates.mean(axis=0).tolist(),
            "mean_estimate_Monte_Carlo_SE": (estimates.std(axis=0, ddof=1)/np.sqrt(args.replications)).tolist(),
            "empirical_SD_near_far": estimates.std(axis=0, ddof=1).tolist(),
            "mean_CR2_SE_near_far": se.mean(axis=0).tolist(),
            "observed_probability_range_without_clipping": probability_support,
            "full_reference_numeric_comparisons": len(errors), "maximum_reference_error": max(errors),
        }
        for key, value in all_results.items():
            arrays[name+"__"+key] = value
        print(json.dumps({"case_complete": name, **report["cases"][name]}), flush=True)
    report["elapsed_seconds"] = time.monotonic()-start
    (args.output_dir/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(args.output_dir/"SIMULATION_SUMMARIES.npz", **arrays)
    print("Completed synthetic joint-null stress:", args.output_dir)


if __name__ == "__main__":
    main()
