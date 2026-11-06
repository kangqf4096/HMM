"""Dependency-free SVG: IoU, Rand index and detected/true change points against K (log axis), small multiples."""
import math
from html import escape

SERIES = (("pd_m100", "#2a78d6", "circle"), ("pd_m400", "#eb6834", "square"), ("pd_fe", "#1baf7a", "triangle"),
          ("seedbs", "#eda100", "diamond"))
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
PANELS = (("IoU", "iou", (0., 1.)), ("Rand index (论文式)", "rand_paper", (0., 1.)),
          ("检出 / 真值切换点数", "cp", (0., None)))


def marker(kind, x, y, color, r=4.5):
    ring = f'stroke="{SURFACE}" stroke-width="2" fill="{color}"'
    if kind == "square":
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" {ring}/>'
    if kind == "triangle":
        return f'<polygon points="{x:.1f},{y - r - 1:.1f} {x - r - 1:.1f},{y + r:.1f} {x + r + 1:.1f},{y + r:.1f}" {ring}/>'
    if kind == "diamond":
        return f'<polygon points="{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y:.1f} {x:.1f},{y + r + 1:.1f} {x - r - 1:.1f},{y:.1f}" {ring}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {ring}/>'


def points(report, method, metric):
    out = []
    for tier in report["tiers"].values():
        if tier["block"] not in ("paper", "alt"):
            continue
        cell = tier["methods"].get(method)
        k = cell and cell["true_changepoints"]
        if not k:
            continue
        if metric == "cp":
            value = cell["predicted_changepoints"] / k if cell["predicted_changepoints"] is not None else None
        else:
            value = (cell[metric] or {}).get("mean")
        if value is not None:
            out.append((k, value))
    return sorted(out)


def render_curves(report, series=SERIES):
    data = {m: {metric: points(report, m, metric) for _, metric, _ in PANELS} for m, _, _ in series}
    ks = sorted({k for m in data.values() for pts in m.values() for k, _ in pts})
    if not ks:
        return None
    width, panel_w, panel_h, left, top = 1240, 330, 250, 60, 70
    lo, hi = math.log(min(ks)), math.log(max(ks))
    span = (hi - lo) or 1.
    height = top + panel_h + 90
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
           f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
           f'<style>text{{font:12px sans-serif;fill:{MUTED}}} .t{{font-weight:600;fill:{INK};font-size:13px}}</style>',
           f'<text x="20" y="28" class="t">按切换点数 K 的性能（论文 Setting 2–4 与交替段，K 为对数轴）</text>']
    for i, (title, metric, (ylo, yhi)) in enumerate(PANELS):
        x0 = left + i * (panel_w + 60)
        values = [v for m in data.values() for _, v in m[metric]]
        top_value = yhi if yhi is not None else max([1.] + values) * 1.1

        def X(k):
            return x0 + panel_w * (math.log(k) - lo) / span

        def Y(v):
            return top + panel_h * (1 - (v - ylo) / (top_value - ylo))
        out.append(f'<text x="{x0}" y="{top - 14}" class="t">{escape(title)}</text>')
        for step in range(5):
            v = ylo + (top_value - ylo) * step / 4
            out.append(f'<line x1="{x0}" x2="{x0 + panel_w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
            out.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        for k in ks:
            out.append(f'<text x="{X(k):.1f}" y="{top + panel_h + 18}" text-anchor="middle">{k:g}</text>')
        out.append(f'<text x="{x0 + panel_w / 2}" y="{top + panel_h + 38}" text-anchor="middle">K</text>')
        for method, color, kind in series:
            pts = data[method][metric]
            if not pts:
                continue
            path = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in pts)
            out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
            out.extend(marker(kind, X(k), Y(v), color) for k, v in pts)
    legend_x = left
    for method, color, kind in series:
        out.append(marker(kind, legend_x, height - 24, color))
        out.append(f'<text x="{legend_x + 12}" y="{height - 20}">{escape(method)}</text>')
        legend_x += 130
    out.append("</svg>")
    return "\n".join(out) + "\n"
