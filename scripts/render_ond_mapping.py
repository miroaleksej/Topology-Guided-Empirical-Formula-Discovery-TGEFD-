import csv
from pathlib import Path


def main() -> int:
    in_path = Path("docs/benchmarks/ond_class_mapping.csv")
    out_path = Path("docs/benchmarks/ond_class_mapping.svg")
    rows = []
    with in_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            rows.append(row)

    col_widths = [220, 220, 280, 300]
    row_height = 36
    header_height = 44
    width = sum(col_widths) + 40
    height = header_height + row_height * len(rows) + 40
    x0 = 20
    y0 = 20

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white' />",
        f"<text x='{width/2:.1f}' y='{y0 - 2}' text-anchor='middle' font-size='18' font-family='Arial'>A/B/C → OND I–IV Mapping</text>",
    ]

    # Header background
    svg.append(
        f"<rect x='{x0}' y='{y0 + 8}' width='{sum(col_widths)}' height='{header_height - 8}' fill='#f2f4f8' stroke='#cbd2d9' />"
    )

    # Header text
    x = x0
    for idx, header in enumerate(headers):
        svg.append(
            f"<text x='{x + 8}' y='{y0 + 34}' font-size='12' font-family='Arial' fill='#111'>{header}</text>"
        )
        x += col_widths[idx]

    # Rows
    y = y0 + header_height
    for row_idx, row in enumerate(rows):
        fill = "#ffffff" if row_idx % 2 == 0 else "#f9fafb"
        svg.append(
            f"<rect x='{x0}' y='{y}' width='{sum(col_widths)}' height='{row_height}' fill='{fill}' stroke='#e1e6eb' />"
        )
        x = x0
        for col_idx, cell in enumerate(row):
            svg.append(
                f"<text x='{x + 8}' y='{y + 23}' font-size='11' font-family='Arial' fill='#111'>{cell}</text>"
            )
            x += col_widths[col_idx]
        y += row_height

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
