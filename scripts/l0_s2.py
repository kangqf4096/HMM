"""S2-stratified pivot-law check on an L0 positions build, by replaying the frozen model.

The calibration mismatch (per-S2-bucket PIT KS 0.22-0.29) was measured when calibration had
8 articles and 37 positions in the tight bucket. L0 now fits f1 on 50 calibration articles, so
the same check has to be re-run at the larger n before that mismatch is called structural.

What this does, per requested article: replay P_t (the model's next-token distribution), then
report the analytic PIT F_{1,P_t}(Y_t) inside S2 buckets. Under a correct replay every PIT is
U(0,1), so the KS distance per bucket is the same statistic the old pivot report quoted.

Requires torch + transformers and the frozen local model (GPU by default). Truth is never read
for the statistic; only the stored Y and the replayed P are used.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_hmm.io import new_output, read_json, read_jsonl, write_json
from opt_hmm.paths import project_path
from opt_hmm.pivot_diagnostics import S2_EDGES, pivot_law, probability_summary, uniform_distance


def bucket_label(group):
    low, high = S2_EDGES[group], min(S2_EDGES[group + 1], 1.)
    return f"[{low:g}, {high:g}]"


def group_of(s2):
    return min(max(int(np.searchsorted(S2_EDGES, s2, side="right") - 1), 0), len(S2_EDGES) - 2)


def table(records):
    """Per role x S2 bucket: position count, articles touched, KS distance, mean PIT."""
    cells = defaultdict(list)
    for record in records:
        cells[(record["split"], group_of(record["s2"]))].append(record)
    rows = []
    for (split, group), items in sorted(cells.items()):
        pit = [r["cdf"] for r in items]
        rows.append({"split": split, "s2_interval": bucket_label(group), "positions": len(items),
                     "articles": len({r["article_id"] for r in items}),
                     "ks_to_uniform": uniform_distance(pit), "mean_pit": float(np.mean(pit)),
                     "mean_s2": float(np.mean([r["s2"] for r in items])),
                     "critical_5pct": float(1.36 / np.sqrt(len(items)))})
    summary = []
    for split in sorted({r["split"] for r in records}):
        pit = [r["cdf"] for r in records if r["split"] == split]
        summary.append({"split": split, "positions": len(pit),
                        "articles": len({r["article_id"] for r in records if r["split"] == split}),
                        "ks_to_uniform": uniform_distance(pit), "mean_pit": float(np.mean(pit)),
                        "critical_5pct": float(1.36 / np.sqrt(len(pit)))})
    return rows, summary


def render(rows, summary, meta):
    lines = ["# L0 的 S2 分桶 PIT 检查（模型概率重放）", "",
             f"- 数据目录：`{meta['positions']}`；sources：`{meta['sources']}`；角色：{meta['roles']}",
             f"- 文章 {meta['articles']} 篇、位置 {meta['positions_count']} 个；device={meta['device']}",
             "- PIT = F_{1,P_t}(Y_t) 应在 (0,1) 上均匀；`critical_5pct` 是 KS 的 5% 临界值 1.36/√n。", "",
             "## 1. 按角色汇总", "",
             "| 角色 | 位置 | 文章 | KS | 均值 PIT | 5% 临界 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['split']} | {row['positions']} | {row['articles']} | "
                     f"{row['ks_to_uniform']:.4f} | {row['mean_pit']:.4f} | {row['critical_5pct']:.4f} |")
    lines += ["", "## 2. 按 S2 分桶", "",
              "| 角色 | S2 区间 | 位置 | 文章 | KS | 均值 PIT | 均值 S2 | 5% 临界 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['s2_interval']} | {row['positions']} | {row['articles']} | "
                     f"{row['ks_to_uniform']:.4f} | {row['mean_pit']:.4f} | {row['mean_s2']:.4f} | "
                     f"{row['critical_5pct']:.4f} |")
    lines += ["", "读法：KS 超过 5% 临界值才算该桶与均匀分布有可检出的偏离；低于临界值时，",
              "原先 8 篇文章上报出的 0.22-0.29 应改判为样本量不足，而不是“结构性失配”。", ""]
    return "\n".join(lines)


def run(positions_dir, output, roles=("calibration",), device="cuda", model=None, limit=None):
    from opt_hmm.pivot_experiment import ModelReplay

    positions_dir = Path(project_path(positions_dir))
    manifest = read_json(positions_dir / "manifest.json")
    sources_dir = Path(project_path(manifest["sources"]))
    metadata = read_json(sources_dir / "metadata.json")
    rows_by_article = defaultdict(list)
    for row in read_jsonl(positions_dir / "positions.jsonl"):
        rows_by_article[row["article_id"]].append(row)
    # Final (possibly resampled) token ids live in articles.jsonl, not in sources.jsonl.
    tokens_by_article = {a["article_id"]: a["token_ids"] for a in read_jsonl(positions_dir / "articles.jsonl")}

    sources = [s for s in read_jsonl(sources_dir / "sources.jsonl") if s.get("split") in set(roles)]
    if not sources:
        raise ValueError(f"sources 里没有角色 {roles} 的文章。")
    if limit:
        sources = sources[:limit]
    model_path = str(project_path(model or metadata["model"]))
    replay = ModelReplay(model_path, metadata, device)

    records = []
    for number, source in enumerate(sources, 1):
        print(f"重放 {number}/{len(sources)}: {source['article_id']}", flush=True)
        probabilities = replay(source["prompt_ids"], tokens_by_article[source["article_id"]])
        for row in rows_by_article[source["article_id"]]:
            p = probabilities[row["t"]]
            records.append({"article_id": source["article_id"], "split": source["split"],
                            "scenario": source["scenario"], "t": row["t"], "y": row["Y"],
                            "s2": probability_summary(p)["s2"],
                            "cdf": pivot_law(p, row["Y"])["cdf"]})
        del probabilities

    rows, summary = table(records)
    meta = {"positions": str(positions_dir), "sources": str(sources_dir), "roles": list(roles),
            "articles": len(sources), "positions_count": len(records), "device": device,
            "model": model_path, "s2_edges": S2_EDGES.tolist()}
    out = new_output(project_path(output))
    write_json(out / "report.json", {"meta": meta, "buckets": rows, "roles": summary, "records": records})
    (out / "report.md").write_text(render(rows, summary, meta), encoding="utf-8")
    return {"report": str(out / "report.md"), "articles": len(sources), "positions": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True, help="l0-build 输出目录（含 manifest/articles/positions）")
    parser.add_argument("--output", required=True, help="新的输出目录，必须为空")
    parser.add_argument("--roles", default="calibration", help="逗号分隔，默认只跑 calibration")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", default=None, help="覆盖 metadata 里的模型路径")
    parser.add_argument("--limit", type=int, default=None, help="只重放前 N 篇，用于快速 smoke")
    args = parser.parse_args()
    result = run(args.positions, args.output, tuple(r.strip() for r in args.roles.split(",") if r.strip()),
                 args.device, args.model, args.limit)
    print(f"report: {result['report']}（{result['articles']} 篇 / {result['positions']} 个位置）")


if __name__ == "__main__":
    main()
