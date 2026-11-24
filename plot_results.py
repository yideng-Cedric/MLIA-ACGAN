"""Plot the archived two-seed AC-GAN evaluation results.

Run from the repository root: python3 plot_results.py
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METRICS = (
    ("FID", "frechet_inception_distance", 250, "lower is better", "{:.1f}"),
    ("KID", "kernel_inception_distance_mean", 0.14, "lower is better", "{:.3f}"),
    ("Inception Score", "inception_score_mean", 5, "higher is better", "{:.2f}"),
    ("Class accuracy", "balanced_accuracy", 1, "higher is better", "{:.1%}"),
)
METHODS = (("original_acgan", "Original AC-GAN"), ("diffaugment_cbn_ema", "DiffAugment + CBN + EMA"))
SEED_COLORS = ("#3976BD", "#E08B45")


def font(size, bold=False):
    candidates = (
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def load_results(paths):
    results = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        by_method = {model["method"]: model for model in report["models"]}
        missing = {name for name, _ in METHODS} - by_method.keys()
        if missing:
            raise ValueError(f"{path}: missing models {sorted(missing)}")
        results.append((path.parent.name, by_method))
    if not results:
        raise ValueError("No metrics.json files found")
    return results


def value(model, key):
    if key == "balanced_accuracy":
        return model["independent_generated"][key]
    return model["metrics"][key]


def draw_panel(draw, box, metric, results):
    title, key, limit, direction, fmt = metric
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill="#FFFFFF", outline="#DBE2EA", width=2)
    draw.text((x0 + 32, y0 + 25), title, font=font(29, True), fill="#17283A")
    draw.text((x1 - 32, y0 + 32), direction.upper(), anchor="ra", font=font(15, True), fill="#607080")

    left, right = x0 + 135, x1 - 65
    top, bottom = y0 + 102, y1 - 79
    centers = (left + (right - left) * 0.22, left + (right - left) * 0.78)

    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = bottom - (bottom - top) * fraction
        draw.line((left, y, right, y), fill="#E9EEF3", width=2)
        draw.text((left - 14, y), fmt.format(limit * fraction), anchor="rm", font=font(15), fill="#72808E")

    def ypos(number):
        return bottom - (bottom - top) * number / limit

    for index, (seed, models) in enumerate(results):
        offset = (index - (len(results) - 1) / 2) * 32
        points = [(centers[j] + offset, ypos(value(models[method], key)))
                  for j, (method, _) in enumerate(METHODS)]
        draw.line((*points[0], *points[1]), fill=SEED_COLORS[index % len(SEED_COLORS)], width=3)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=SEED_COLORS[index % len(SEED_COLORS)], outline="white", width=2)

    means = [sum(value(models[method], key) for _, models in results) / len(results) for method, _ in METHODS]
    for x, mean in zip(centers, means):
        y = ypos(mean)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#17283A", outline="white", width=3)
        label_y = max(top + 16, y - 25)
        draw.text((x, label_y), fmt.format(mean), anchor="mb", font=font(19, True), fill="#17283A")

    for x, (_, label) in zip(centers, METHODS):
        if label == "Original AC-GAN":
            draw.text((x, bottom + 24), label, anchor="mt", font=font(16, True), fill="#34465A")
        else:
            draw.text((x, bottom + 20), "DiffAugment +", anchor="mt", font=font(16, True), fill="#34465A")
            draw.text((x, bottom + 40), "CBN + EMA", anchor="mt", font=font(16, True), fill="#34465A")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Evaluation metrics.json files")
    parser.add_argument("--output", type=Path, default=Path("results/metrics_comparison.png"))
    args = parser.parse_args()
    inputs = args.inputs or sorted(Path("results").glob("seed*/metrics.json"))
    results = load_results(inputs)

    image = Image.new("RGB", (1600, 1090), "#F4F7FA")
    draw = ImageDraw.Draw(image)
    draw.text((70, 52), "AC-GAN result comparison", font=font(43, True), fill="#17283A")
    draw.text((72, 111), "SYSU-Shape · 64 px · archived evaluation runs", font=font(23), fill="#526578")

    legend_x = 1050
    for index, (seed, _) in enumerate(results):
        x = legend_x + 150 * index
        draw.ellipse((x, 78, x + 14, 92), fill=SEED_COLORS[index % len(SEED_COLORS)])
        draw.text((x + 22, 85), seed, anchor="lm", font=font(18), fill="#34465A")
    draw.ellipse((legend_x, 113, legend_x + 18, 131), fill="#17283A")
    draw.text((legend_x + 26, 122), "mean across seeds", anchor="lm", font=font(18), fill="#34465A")

    boxes = ((70, 170, 780, 565), (820, 170, 1530, 565),
             (70, 590, 780, 985), (820, 590, 1530, 985))
    for box, metric in zip(boxes, METRICS):
        draw_panel(draw, box, metric, results)

    draw.text((72, 1024), "Points and lines show individual seeds. Dark markers and labels show the two-seed mean.",
              font=font(18), fill="#526578")
    draw.text((72, 1051), "FID / KID / IS: 256 generated images per run; class accuracy: 1,280 generated images per run.",
              font=font(17), fill="#526578")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
