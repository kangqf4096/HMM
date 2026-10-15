"""Dependency-free diagnostic SVG; always show the first test document."""
from html import escape

import numpy as np

from .detector import intervals


def render_example(record, predictions, path):
    selected = [p for p in predictions if p["id"] == record["id"]]
    width, left, right = 1100, 150, 1080
    height = 110 + 125 * len(selected)
    length = len(record["z"])
    def x(t):
        return left + (right - left) * t / max(length, 1)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font:13px sans-serif;fill:#202631} .name{font-weight:600}</style>',
                f'<text x="18" y="25" class="name">First test document: {escape(record["id"])}</text>',
                '<text x="18" y="48">Blue: score; red: null threshold; orange: output; green: true signal; gray: unavailable</text>',
                '<text x="18" y="77" class="name">Truth Z</text>']
    truth, mask = np.asarray(record["z"], bool), np.asarray(record.get("mask", [True] * length), bool)
    for start, stop in intervals(truth & mask):
        elements.append(f'<rect x="{x(start):.2f}" y="62" width="{x(stop)-x(start):.2f}" height="18" fill="#16804a"/>')
    for start, stop in intervals(~mask):
        elements.append(f'<rect x="{x(start):.2f}" y="62" width="{x(stop)-x(start):.2f}" height="18" fill="#9ca3af"/>')
    for i, prediction in enumerate(selected):
        top = 105 + i * 125
        bottom = top + 65
        elements.extend([f'<text x="18" y="{top+20}" class="name">{escape(prediction["method"])}</text>',
                         f'<line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="#ccd2d8"/>'])
        threshold_y = bottom - prediction["threshold"] * 65
        elements.append(f'<line x1="{left}" x2="{right}" y1="{threshold_y:.2f}" y2="{threshold_y:.2f}" stroke="#c53d3d" stroke-dasharray="4 3"/>')
        points = " ".join(f"{x(t+.5):.2f},{bottom-65*s:.2f}" for t, s in enumerate(prediction["scores"]))
        elements.append(f'<polyline points="{points}" fill="none" stroke="#225ec2" stroke-width="1.5"/>')
        for start, stop in intervals(prediction["labels"]):
            elements.append(f'<rect x="{x(start):.2f}" y="{bottom+7}" width="{x(stop)-x(start):.2f}" height="9" fill="#d97706"/>')
        elements.extend([f'<text x="{left}" y="{bottom+33}">0</text>',
                         f'<text x="{right-90}" y="{bottom+33}">{length} tokens</text>'])
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
