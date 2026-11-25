"""Plot the two AC-GAN recipes from a matched scores CSV or archived metrics JSON."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter


RESULTS = Path(__file__).resolve().parent / "results"
METRICS = (
    ("FID ↓", "fid", "frechet_inception_distance", 1, ".1f"),
    ("KID ↓", "kid", "kernel_inception_distance_mean", 1, ".4f"),
    ("IS ↑", "is", "inception_score_mean", 1, ".2f"),
    ("Class hit rate ↑", "class_hit_rate", "balanced_accuracy", 100, ".1f"),
)
COLORS = ("#B8BEC5", "#285879")


def load_pair(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as source:
            overall = [row for row in csv.DictReader(source)
                       if row["category"] == "overall" and float(row.get("truncation", 0)) == 0]
        plain = [row for row in overall if row["method"] == "plain_acgan"]
        improved = [row for row in overall
                    if row["method"] == "conditional" and float(row["horizontal_flip_prob"]) == 0.25]
        if len(plain) != 1 or len(improved) != 1:
            raise ValueError(f"{path}: expected one plain and one improved AC-GAN at 25% horizontal flip")
        for key in ("seed", "step", "framing", "n_real", "n_generated"):
            if plain[0][key] != improved[0][key]:
                raise ValueError(f"{path}: AC-GAN rows differ in {key}")
        return tuple([float(row[key]) for _, key, _, _, _ in METRICS]
                     for row in (plain[0], improved[0]))

    report = json.loads(path.read_text(encoding="utf-8"))
    models = {entry["method"]: entry for entry in report["models"]}
    pair = []
    for name in ("original_acgan", "diffaugment_cbn_ema"):
        model = models[name]
        pair.append([model["metrics"][key] if key != "balanced_accuracy"
                     else model["independent_generated"][key] for _, _, key, _, _ in METRICS])
    return tuple(pair)


def render(pairs, labels, outputs):
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#606060",
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }):
        figure, axes = plt.subplots(1, len(METRICS), figsize=(7.05, 2.16))
        figure.subplots_adjust(left=0.075, right=0.975, bottom=0.20, top=0.84, wspace=0.55)

        for index, (axis, (title, _, _, scale, fmt)) in enumerate(zip(axes, METRICS)):
            values = [[pair[method][index] * scale for pair in pairs] for method in range(2)]
            centers = [0, 1]
            averages = [mean(column) for column in values]
            unit = "%" if scale == 100 else ""
            axis.bar(centers, averages, color=COLORS, width=0.56, zorder=2)
            for position, average, column in zip(centers, averages, values):
                if len(column) > 1:
                    axis.scatter([position] * len(column), column, s=11, facecolors="white",
                                 edgecolors="#202020", linewidths=0.6, zorder=3)
                axis.annotate(format(average, fmt) + unit, (position, max(column)), xytext=(0, 4),
                              textcoords="offset points", ha="center", va="bottom", fontsize=8)
            axis.set_title(title, fontsize=9.2, fontweight="bold", pad=8)
            axis.set_xlim(-0.50, 1.50)
            axis.set_ylim(0, 100 if scale == 100 else max(map(max, values)) * 1.26)
            axis.set_xticks([])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            if unit:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            axis.tick_params(axis="y", labelsize=7.5, length=2.5, width=0.6, pad=2)
            axis.grid(axis="y", color="#E1E3E6", linewidth=0.55, zorder=0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        handles = [Patch(facecolor=color, label=label) for color, label in zip(COLORS, labels)]
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                      ncol=2, frameon=False, fontsize=8.5, handlelength=1.25, columnspacing=1.8)
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=400, facecolor="white")
            print(output)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Matched scores CSVs or archived metrics JSONs")
    parser.add_argument("--output", type=Path, help="Write a single PNG or PDF instead of both defaults")
    args = parser.parse_args()
    inputs = args.inputs or [RESULTS / "matched_acgan" / "scores.csv"]
    formats = {path.suffix.lower() for path in inputs}
    if formats == {".csv"}:
        labels = ("Plain AC-GAN", "Improved AC-GAN")
    elif formats == {".json"}:
        labels = ("Early AC-GAN", "DiffAugment + CBN + EMA")
    else:
        parser.error("Use matched CSVs or archived JSONs from the same evaluation protocol.")
    outputs = [args.output] if args.output else [RESULTS / f"metrics_comparison.{fmt}" for fmt in ("png", "pdf")]
    render([load_pair(path) for path in inputs], labels, outputs)


if __name__ == "__main__":
    main()
