"""Stages A/B: independent calibration, synthetic sequences, complete null refits."""
import numpy as np

from .emissions import sample_signal, theoretical_density
from .experiment import run_evaluation
from .io import new_output, write_jsonl


def sample_states(rng, n, epsilon, rho):
    z = np.empty(n, dtype=bool)
    z[0] = rng.random() < epsilon
    for t in range(1, n):
        z[t] = z[t - 1] if rng.random() < rho else rng.random() < epsilon
    return z


def run(config, output):
    output = new_output(output)
    n = int(config["tokens"])
    count = int(config["test_documents"])
    if n < 4 or count < 1 or min(config["null_cal"], config["null_test"], config["cal_pivots"]) < 1:
        raise ValueError("Invalid simulation sample sizes.")
    streams = np.random.SeedSequence(config.get("seed", 20261015)).spawn(4)
    cal_rng, nc_rng, nt_rng, test_rng = [np.random.default_rng(s) for s in streams]
    p = config.get("signal_p", [.5, .5])
    calibration_p = config.get("calibration_p", p)
    epsilon, rho = config.get("epsilon", .3), config.get("rho", .95)
    if not 0 <= epsilon <= 1 or not 0 <= rho < 1:
        raise ValueError("Invalid state parameters.")
    records = []
    def record(identifier, role, y, z, variant):
        return {"id": identifier, "group": identifier, "role": role, "variant": variant,
                "y": y, "z": z, "mask": np.ones(len(y), bool)}
    y_cal = sample_signal(cal_rng, config["cal_pivots"], calibration_p)
    records.append(record("cal", "calibration", y_cal, np.ones(len(y_cal), bool), "watermarked"))
    for role, number, rng in (("null_cal", config["null_cal"], nc_rng),
                              ("null_test", config["null_test"], nt_rng)):
        for i in range(number):
            y = np.clip(rng.random(n), np.nextafter(0., 1.), np.nextafter(1., 0.))
            records.append(record(f"{role}-{i}", role, y, np.zeros(n, bool), "null"))
    scenario = config.get("scenario", "markov")
    for i in range(count):
        if scenario == "markov":
            z = sample_states(test_rng, n, epsilon, rho)
        elif scenario == "iid":
            z = test_rng.random(n) < epsilon
        elif scenario == "island":
            length = min(n, int(config.get("segment_length", 40)))
            if length < 1:
                raise ValueError("Segment length must be positive.")
            start = test_rng.integers(n - length + 1)
            z = np.zeros(n, bool)
            z[start:start + length] = True
        elif scenario in ("hole", "strong_weak"):
            z = np.ones(n, bool)
            if scenario == "hole":
                length = min(n, int(config.get("segment_length", 40)))
                start = test_rng.integers(n - length + 1)
                z[start:start + length] = False
        else:
            raise ValueError(f"Unknown scenario {scenario}")
        y = np.clip(test_rng.random(n), np.nextafter(0., 1.), np.nextafter(1., 0.))
        y[z] = sample_signal(test_rng, int(z.sum()), p)
        if scenario == "strong_weak":
            weak = (np.arange(n) // config.get("segment_length", 40)) % 2 == 1
            y[weak] = sample_signal(test_rng, int(weak.sum()), [.95, .05])
        records.append(record(f"test-{i}", "test", y, z, scenario))
    write_jsonl(output / "observations.jsonl", records)
    # Only stationary Markov simulations have the matching oracle model here.
    oracle = (lambda y: theoretical_density(y, p)) if scenario == "markov" else None
    return run_evaluation(records, output, config, oracle,
                          {"epsilon": epsilon, "rho": rho} if oracle else None)
