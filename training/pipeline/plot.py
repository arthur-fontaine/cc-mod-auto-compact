"""Draw the benchmark: right action against latency, one point per model.

    uv run python pipeline/plot.py     # -> ../docs/benchmark.svg

Reads results/bench-*.json from pipeline/benchmark.py. The SVG follows the viewer's light
or dark mode. From Nisev's plot.py.
"""
import json
import math
from pathlib import Path

from benchmark import CONTENDERS, ROOT

OUT = ROOT.parent / "docs" / "benchmark.svg"
W, H = 760, 470
LEFT, RIGHT, TOP, BOTTOM = 72, 28, 56, 64
X_MIN, X_MAX = 100, 4000  # ms, log scale
Y_MIN, Y_MAX = 70, 100  # % right action
# The zone to aim for: under the Doherty threshold (people stay engaged when a system answers
# within 400 ms; Doherty and Thadani, IBM, 1982), and above the lowest score that beats always
# keeping the context by more than chance on these points (benchmark.threshold).
GOAL_MS = 400
LABELS = {"spisev": "below", "qwen3.5-2b": "below", "elephant": "left", "spisev-previous": "below"}


def results():
    out = []
    for name, *_ in CONTENDERS:
        path = ROOT / "results" / f"bench-{name}.json"
        if path.exists():
            out.append(json.loads(path.read_text()))
    return out


def x(ms):
    ms = min(max(ms, X_MIN), X_MAX)
    return LEFT + (math.log10(ms) - math.log10(X_MIN)) / (math.log10(X_MAX) - math.log10(X_MIN)) * (W - LEFT - RIGHT)


def y(pct):
    return TOP + (Y_MAX - pct) / (Y_MAX - Y_MIN) * (H - TOP - BOTTOM)


def svg():
    rs = results()
    keep_pct, goal_pct, n = 100 * rs[0]["baseline_keep"], 100 * rs[0]["threshold"], rs[0]["n"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">',
           """<style>
  .bg { fill: #ffffff; } text { fill: #1f2328; font-size: 13px; } .muted { fill: #656d76; font-size: 12px; }
  .grid { stroke: #d8dee4; } .default { stroke: #656d76; }
  .zone { fill: #dafbe1; stroke: #1a7f37; } text.goal { fill: #1a7f37; }
  .local { fill: #0969da; stroke: #0969da; } .cloud { fill: #ffffff; stroke: #8250df; }
  .local-tail { stroke: #0969da; } .cloud-tail { stroke: #8250df; }
  .previous circle { fill: #ffffff; } .previous { opacity: 0.5; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; } text { fill: #e6edf3; } .muted { fill: #8d96a0; }
    .grid { stroke: #30363d; } .default { stroke: #8d96a0; }
    .zone { fill: #12261e; stroke: #3fb950; } text.goal { fill: #3fb950; }
    .local { fill: #4493f8; stroke: #4493f8; } .cloud { fill: #0d1117; stroke: #ab7df8; }
    .local-tail { stroke: #4493f8; } .cloud-tail { stroke: #ab7df8; }
    .previous circle { fill: #0d1117; }
  }
</style>""",
           f'<rect class="bg" width="{W}" height="{H}"/>']

    gx, gy = x(GOAL_MS), y(goal_pct)
    out.append(f'<rect x="{LEFT}" y="{TOP}" width="{gx - LEFT:.1f}" height="{gy - TOP:.1f}" class="zone" '
               'stroke-dasharray="4 3"/>')
    out.append(f'<text x="{LEFT + 10}" y="{gy - 30:.1f}" class="goal" style="font-weight: 600">Where we want to be</text>')
    out.append(f'<text x="{LEFT + 10}" y="{gy - 13:.1f}" class="goal" style="font-size: 12px">fast and accurate</text>')
    # Both edges are explained under the zone: the Qwen3.5 point sits right on the latency edge.
    out.append(f'<text x="{LEFT + 10}" y="{gy + 15:.1f}" class="goal" style="font-size: 11px">'
               f'≥ {goal_pct:.1f}%: beats always keeping, p &lt; 0.05</text>')
    out.append(f'<text x="{LEFT + 10}" y="{gy + 29:.1f}" class="goal" style="font-size: 11px">'
               f'&lt; {GOAL_MS} ms: the Doherty threshold</text>')

    for ms in (100, 200, 500, 1000, 2000):
        out.append(f'<line class="grid" x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="{TOP}" y2="{H - BOTTOM}"/>')
        out.append(f'<text class="muted" x="{x(ms):.1f}" y="{H - BOTTOM + 18}" text-anchor="middle">'
                   f'{ms if ms < 1000 else f"{ms // 1000} s"}{" ms" if ms < 1000 else ""}</text>')
    for pct in range(Y_MIN, Y_MAX + 1, 10):
        out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(pct):.1f}" y2="{y(pct):.1f}"/>')
        out.append(f'<text class="muted" x="{LEFT - 8}" y="{y(pct) + 4:.1f}" text-anchor="end">{pct}%</text>')

    out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y(keep_pct):.1f}" y2="{y(keep_pct):.1f}" '
               'class="default" stroke-dasharray="6 4"/>')
    out.append(f'<text class="muted" x="{W - RIGHT - 6}" y="{y(keep_pct) + 16:.1f}" text-anchor="end">'
               f'always keep the context ({keep_pct:.1f}%)</text>')

    for r in rs:
        # A model measured on the v1 state (an earlier version): drawn hollow and faded, with a note.
        previous = r.get("previous")
        if previous:
            out.append('<g class="previous">')
        local = r["where"].startswith("Local")
        kind = "local" if local else "cloud"
        lat = r["latency_ms"]
        cx, cy = x(lat["p50"]), y(100 * r["right_action"])
        tail_end = x(lat["p95"])
        out.append(f'<line x1="{cx:.1f}" x2="{tail_end:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="{kind}-tail" '
                   'stroke-width="2" stroke-opacity="0.35"/>')
        if lat["p95"] > X_MAX:  # the tail runs off the chart: say how far
            out.append(f'<text class="muted" x="{tail_end - 4:.1f}" y="{cy - 6:.1f}" text-anchor="end" '
                       f'style="font-size: 11px">p95 {lat["p95"] / 1000:.1f} s →</text>')
        out.append(f'<circle class="{kind}" cx="{cx:.1f}" cy="{cy:.1f}" r="6" stroke-width="2.5"/>')
        label = r["label"].split(" (")[0] + (" (local)" if local else " (cloud)")
        # Spisev, Elephant and Qwen3.5 sit within a point of each other: spread their labels.
        place = LABELS.get(r["name"], "above")
        lx, ly, anchor = {"above": (cx, cy - 12, "middle"), "below": (cx, cy + 24, "middle"),
                          "left": (cx - 10, cy - 10, "end")}[place]
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" style="font-weight: 600">{label}</text>')
        if previous:
            out.append(f'<text x="{lx:.1f}" y="{ly + 14:.1f}" text-anchor="{anchor}" class="muted" '
                       f'style="font-size: 11px">{previous}</text>')
            out.append("</g>")

    out.append(f'<text x="{LEFT}" y="26" style="font-size: 16px; font-weight: 600">Right action against latency</text>')
    out.append(f'<text x="{(LEFT + W - RIGHT) / 2:.0f}" y="{H - 18}" text-anchor="middle" class="muted">'
               'Latency per decision: median, line to p95 (log scale)</text>')
    out.append(f'<text transform="translate(18 {(TOP + H - BOTTOM) / 2:.0f}) rotate(-90)" text-anchor="middle" '
               f'class="muted">Right action, {n} held-out decisions</text>')

    lx = W - RIGHT - 230
    out.append(f'<circle class="local" cx="{lx}" cy="21" r="5" stroke-width="2"/>')
    out.append(f'<text x="{lx + 10}" y="25" class="muted">Local (M4 Pro, llama.cpp)</text>')
    out.append(f'<circle class="cloud" cx="{lx + 170}" cy="21" r="5" stroke-width="2"/>')
    out.append(f'<text x="{lx + 180}" y="25" class="muted">Cloud</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(svg())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
