"""Compare exact SeedBS backends on identical data/seeds, without changing run outputs."""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opt_hmm.l0 import seedbs as sb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Read the first article of this *-pos directory.")
    parser.add_argument("--B", type=int, nargs="+", default=[20])
    parser.add_argument("--T", type=int, default=999)
    parser.add_argument("--Tp", type=int, default=1999)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/sbbench.json")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    fast = sb._load_fast()
    if fast is None:
        parser.error("Install requirements/cpu-fast.txt first.")
    # Warm compilation separately from steady-state timing.
    started = perf_counter()
    sample = np.linspace(.01, .99, 16)[None, :]
    fast.ks_integers(sample)
    fast.ks_exceeds(sample, 1)
    warmup = perf_counter() - started
    print(f"Numba warmup (excluded): {warmup:.3f}s", flush=True)
    if args.input:
        from opt_hmm.l2.data import iter_position_groups
        groups = iter_position_groups(args.input / "positions.jsonl")
        try:
            name, rows = next(groups)
        finally:
            groups.close()
        cases = [(name, np.array([row["Y"] for row in rows]))]
    else:
        rng = np.random.default_rng(27)
        null = rng.random(496)
        splice = null.copy()
        splice[:248] = rng.beta(5, 1, 248)
        cases = [("synthetic_null", null), ("synthetic_splice", splice)]
    results = []
    for name, y in cases:
        for B in args.B:
            timings = {backend: [] for backend in ("numpy", "numba")}
            for repeat in range(args.repeat):
                outputs = {}
                order = ("numpy", "numba") if repeat % 2 == 0 else ("numba", "numpy")
                for backend in order:
                    os.environ["OPT_HMM_SEEDBS_BACKEND"] = backend
                    wall, cpu = perf_counter(), process_time()
                    pre = sb.seedbs_precompute(y, B, 17, T=args.T, Tp=args.Tp)
                    measured = {"wall_seconds": perf_counter() - wall, "cpu_seconds": process_time() - cpu}
                    timings[backend].append(measured)
                    outputs[backend] = pre
                    print(f"{name} n={len(y)} B={B} {backend}: {measured}", flush=True)
                for key in outputs["numpy"]:
                    if key != "seconds":
                        np.testing.assert_equal(outputs["numpy"][key], outputs["numba"][key], err_msg=key)
            wall = {k: statistics.median(v["wall_seconds"] for v in values) for k, values in timings.items()}
            row = {"article": name, "n": len(y), "B": B, "exact_match": True,
                   "timings": timings, "wall_speedup": wall["numpy"] / wall["numba"]}
            results.append(row)
            print(f"Exact match; speedup {row['wall_speedup']:.2f}x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"numpy": np.__version__, "warmup_seconds": warmup,
                                      "T": args.T, "Tp": args.Tp, "results": results}, indent=2) + "\n")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
