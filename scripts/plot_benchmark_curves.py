import argparse
import json
import math
from pathlib import Path
from typing import Iterable


COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]


def _scale(values: Iterable[float], out_min: float, out_max: float):
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        def _f(_v: float) -> float:
            return (out_min + out_max) * 0.5
        return _f

    def _f(v: float) -> float:
        return out_min + (v - vmin) / (vmax - vmin) * (out_max - out_min)

    return _f


def _resolve_metric(payload: dict, metric_override: str | None) -> tuple[str, list[dict]]:
    if "metrics" in payload:
        metrics = payload["metrics"]
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("metrics must be a non-empty mapping")
        metric_name = metric_override or payload.get("default_metric") or next(iter(metrics.keys()))
        if metric_name not in metrics:
            raise ValueError(f"metric '{metric_name}' not found in metrics")
        series = metrics[metric_name]
        return metric_name, series
    metric_name = payload.get("metric", "Quality")
    series = payload.get("series", [])
    return metric_name, series


def _render_svg(payload: dict, output_path: Path, metric_override: str | None) -> None:
    width = 900
    height = 520
    margin_left = 70
    margin_right = 50
    margin_top = 70
    margin_bottom = 70

    n_values = payload["n_values"]
    metric, series = _resolve_metric(payload, metric_override)
    title = payload.get("title", "Detection quality vs N")

    log_n = [math.log2(float(n)) for n in n_values]
    x_scale = _scale(log_n, margin_left, width - margin_right)
    y_scale = _scale([0.0, 1.0], height - margin_bottom, margin_top)

    def x_pos(n: float) -> float:
        return x_scale(math.log2(n))

    def y_pos(v: float) -> float:
        return y_scale(v)

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2:.1f}' y='{margin_top/2:.1f}' text-anchor='middle' font-size='20' font-family='Arial'>{title}</text>",
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#222' stroke-width='1' />",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#222' stroke-width='1' />",
    ]

    for n in n_values:
        x = x_pos(n)
        svg.append(
            f"<line x1='{x:.1f}' y1='{height - margin_bottom}' x2='{x:.1f}' y2='{height - margin_bottom + 6}' stroke='#222' stroke-width='1' />"
        )
        svg.append(
            f"<text x='{x:.1f}' y='{height - margin_bottom + 24}' text-anchor='middle' font-size='12' font-family='Arial'>{int(n)}</text>"
        )

    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(v)
        svg.append(
            f"<line x1='{margin_left - 6}' y1='{y:.1f}' x2='{margin_left}' y2='{y:.1f}' stroke='#222' stroke-width='1' />"
        )
        svg.append(
            f"<text x='{margin_left - 12}' y='{y + 4:.1f}' text-anchor='end' font-size='12' font-family='Arial'>{v:.2f}</text>"
        )
        svg.append(
            f"<line x1='{margin_left}' y1='{y:.1f}' x2='{width - margin_right}' y2='{y:.1f}' stroke='#eee' stroke-width='1' />"
        )

    svg.append(
        f"<text x='{width/2:.1f}' y='{height - 18}' text-anchor='middle' font-size='13' font-family='Arial'>N (log2 scale)</text>"
    )
    svg.append(
        f"<text x='18' y='{height/2:.1f}' text-anchor='middle' font-size='13' font-family='Arial' transform='rotate(-90 18 {height/2:.1f})'>{metric}</text>"
    )

    for idx, series_entry in enumerate(series):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        values = series_entry["values"]
        points = " ".join(
            f"{x_pos(n):.1f},{y_pos(v):.1f}" for n, v in zip(n_values, values)
        )
        svg.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}' />"
        )
        for n, v in zip(n_values, values):
            svg.append(
                f"<circle cx='{x_pos(n):.1f}' cy='{y_pos(v):.1f}' r='3' fill='{color}' />"
            )

    legend_x = width - margin_right - 200
    legend_y = margin_top + 10
    svg.append(
        f"<rect x='{legend_x}' y='{legend_y}' width='190' height='{len(series) * 20 + 20}' fill='white' stroke='#ccc' />"
    )
    for idx, series_entry in enumerate(series):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        y = legend_y + 20 + idx * 20
        svg.append(
            f"<line x1='{legend_x + 10}' y1='{y - 6}' x2='{legend_x + 30}' y2='{y - 6}' stroke='{color}' stroke-width='2' />"
        )
        svg.append(
            f"<text x='{legend_x + 36}' y='{y - 2}' font-size='12' font-family='Arial'>{series_entry['label']}</text>"
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render benchmark curves to SVG.")
    parser.add_argument("input", nargs="+", help="Path(s) to JSON curve files")
    parser.add_argument("--out-dir", default="docs/benchmarks", help="Output directory for SVGs")
    parser.add_argument("--metric", default=None, help="Metric to plot when JSON contains multiple metrics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.input:
        path = Path(input_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        output_path = out_dir / f"{path.stem}.svg"
        _render_svg(payload, output_path, args.metric)
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
