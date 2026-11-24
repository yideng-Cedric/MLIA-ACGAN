"""Measure held-out SYSU-Shape quality, category control, and diversity."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import svdvals
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from torch.nn.functional import normalize
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_isc import isc_features_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric

from train import (DATA_DIR, image_tensor, load_generation_checkpoint, load_split,
                   sample, sample_noise, save_samples)
from evaluate_cbn import agreement, features, within_class_distance


N_OVERALL = 512
N_CLASS = 256
NOISE_SEED = 56017


def frechet_distance(real, fake):
    # The trace of the covariance geometric mean equals the sum of singular values
    # of the cross-product of the centered and scaled feature matrices.
    a, b = real.numpy().astype(np.float64), fake.numpy().astype(np.float64)
    a_mean, b_mean = a.mean(0), b.mean(0)
    a = (a - a_mean) / np.sqrt(len(a) - 1)
    b = (b - b_mean) / np.sqrt(len(b) - 1)
    return float(max(0.0, np.sum((a_mean - b_mean) ** 2) + np.sum(a ** 2) + np.sum(b ** 2)
                     - 2 * svdvals(a @ b.T).sum()))


def metrics(real, fake, logits, kid_subset_size):
    return {
        "n_real": len(real), "n_generated": len(fake),
        "frechet_inception_distance": frechet_distance(real, fake),
        **kid_features_to_metric(fake, real, kid_subset_size=kid_subset_size, kid_subsets=100,
                                 rng_seed=42, verbose=False),
        **isc_features_to_metric(logits, splits=10, rng_seed=42),
    }


def paste_row(canvas, tiles, x, y):
    for image in tiles:
        tile = Image.fromarray(image.permute(1, 2, 0).numpy())
        tile = tile.resize((96, 64) if tile.width == 96 else (64, 64))
        canvas.paste(tile, (x + (96 - tile.width) // 2, y))
        x += 101


def plot_grid(output, label, classes, panels):
    canvas = Image.new("RGB", (1120, 52 + 100 * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.load_default(size=15)
    bold = ImageFont.load_default(size=21)
    draw.text((14, 10), label, fill="#182531", font=bold)
    for row, (name, tiles) in enumerate(zip(classes, panels)):
        y = 50 + row * 100
        draw.text((14, y + 24), name, fill="#182531", font=regular)
        for x, heading, images in tiles:
            draw.text((x, y - 5), heading, fill="#475561", font=regular)
            paste_row(canvas, images, x - 14, y + 17)
    canvas.save(output)


def conditioning_grid(output, name, classes, real, labels, generated):
    rows = [
        [(115, "Held-out real photos", real[labels == c][:2]),
         (335, "AC-GAN with fixed noise across classes", generated[c][:6])]
        for c in range(len(classes))
    ]
    plot_grid(output, name, classes, rows)


def noise_grid(output, name, classes, real, labels, generated):
    references = torch.stack([real[labels == c][0] for c in range(len(classes))])
    rows = [[(115, "Held-out real photos (one per class)", references)],
            [(115, "Ten fixed noise draws, G(z)", generated[:5])],
            [(115, "Noise draws continued", generated[5:10])]]
    plot_grid(output, name, ["Real", "Generated", "Generated"], rows)


def evaluate(checkpoints, output, data_dir, weights_path, overall_noise_seed=1234,
             class_noise_seed=NOISE_SEED, truncation=0.0):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(data_dir)
    classes = split["classes"]
    train_labels = torch.tensor([label for _, label in split["train"]])
    val_labels = torch.tensor([label for _, label in split["validation"]])
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048", "logits_unbiased"],
        feature_extractor_weights_path=weights_path,
    ).to(device).eval()
    # Use the held-out class frequencies for overall conditional metrics.
    label_index = (torch.arange(N_OVERALL) * len(val_labels)) // N_OVERALL
    metric_labels = val_labels[label_index]
    metric_labels = metric_labels[torch.randperm(N_OVERALL, generator=torch.Generator().manual_seed(42))]
    target = torch.arange(len(classes)).repeat_interleave(N_CLASS)
    cached = {}
    score_rows = []
    conditional_samples = {}
    flip_comparisons = defaultdict(dict)
    plain_comparisons = {}
    output.mkdir(parents=True, exist_ok=True)

    for checkpoint in checkpoints:
        generator, config, stored_classes, step = load_generation_checkpoint(checkpoint, device)
        framing = config.get("framing", "full64")
        conditional = config.get("conditional", True)
        seed = config["seed"]
        flip_prob = config.get("horizontal_flip_prob", 0.0)
        plain_acgan = config.get("plain_acgan", False)
        assert stored_classes == classes
        method = "conditional" if conditional else "unconditional"
        if plain_acgan:
            method = "plain_acgan"
        flip_suffix = f"_hflip{flip_prob:g}" if flip_prob else ""
        folder = output / f"{framing}_{method}{flip_suffix}_seed{seed}_step{step}"
        folder.mkdir(parents=True, exist_ok=True)
        if framing not in cached:
            train_images = torch.stack([image_tensor(path, framing) for path, _ in split["train"]])
            real = torch.stack([image_tensor(path, framing) for path, _ in split["validation"]])
            train_features, _ = features(extractor, train_images, device)
            val_features, _ = features(extractor, real, device)
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
            classifier.fit(normalize(train_features).numpy(), train_labels.numpy())
            cached[framing] = real, val_features, classifier
        real, val_features, classifier = cached[framing]
        val_pred = classifier.predict(normalize(val_features).numpy())
        latent_limit = truncation if conditional else 0.0
        if conditional:
            metric_images = sample(generator, metric_labels, device, noise_seed=overall_noise_seed,
                                   truncation=latent_limit)
        else:
            metric_images = sample_noise(generator, N_OVERALL, device, noise_seed=overall_noise_seed)
        fake_features, logits = features(extractor, metric_images, device)
        report = dict(framing=framing, conditional=conditional, seed=seed, step=step,
                      method=method, horizontal_flip_prob=flip_prob, truncation=latent_limit,
                      checkpoint=str(checkpoint),
                      overall_noise_seed=overall_noise_seed, class_noise_seed=class_noise_seed,
                      classes=classes, overall=metrics(val_features, fake_features, logits, 64),
                      classifier_on_real=agreement(val_pred, val_labels.numpy(), classes))
        if conditional:
            fresh_images = sample(generator, target, device, shared_noise=True,
                                  noise_seed=class_noise_seed, truncation=latent_limit)
            fresh_features, fresh_logits = features(extractor, fresh_images, device)
            prediction = classifier.predict(normalize(fresh_features).numpy())
            report["per_class"] = {
                name: metrics(val_features[val_labels == c], fresh_features[target == c],
                              fresh_logits[target == c], 32)
                for c, name in enumerate(classes)
            }
            report["classifier_on_generated"] = agreement(prediction, target.numpy(), classes)
            report["class_hit_rate"] = {
                name: float((prediction[target.numpy() == c] == c).mean())
                for c, name in enumerate(classes)
            }
            report["feature_diversity_real"] = within_class_distance(val_features, val_labels, len(classes))
            report["feature_diversity_generated"] = within_class_distance(fresh_features, target, len(classes))
            rows = fresh_images.unflatten(0, (len(classes), N_CLASS))
            model_name = "Basic AC-GAN" if plain_acgan else f"AC-GAN | HFlip {flip_prob:.0%}"
            conditioning_grid(folder / "comparison.png", f"{framing} | {model_name} | seed {seed}",
                              classes, real, val_labels, rows)
            if plain_acgan:
                plain_comparisons[seed, step, framing] = rows
            else:
                flip_comparisons[seed, step, framing][flip_prob] = (real, rows)
                conditional_samples[seed, step, framing] = rows
        else:
            prediction = classifier.predict(normalize(fake_features).numpy())
            report["predicted_class_counts"] = dict(zip(
                classes, np.bincount(prediction, minlength=len(classes)).tolist()
            ))
            noise_draws = sample_noise(generator, 10, device, noise_seed=NOISE_SEED)
            noise_grid(folder / "comparison.png", f"{framing} | Unconditional GAN | seed {seed}",
                       classes, real, val_labels, noise_draws)
            if (seed, step, framing) in conditional_samples:
                conditioned = conditional_samples[seed, step, framing]
                panels = [
                    [(115, "Real validation", real[val_labels == c][:1]),
                     (335, "Conditional | two fixed noise draws", conditioned[c][:2])]
                    for c in range(len(classes))
                ]
                panels += [[(115, "Unconditional | five noise draws", noise_draws[:5])],
                           [(115, "Unconditional | five more noise draws", noise_draws[5:])]]
                plot_grid(output / f"conditional_vs_unconditional_seed{seed}_step{step}.png",
                          f"SYSU-Shape | {framing} | Seed {seed}",
                          [*classes, "Generated", "Generated"], panels)
        diversity = report.get("feature_diversity_generated", [])
        class_diversity = dict(zip(classes, diversity))
        if diversity:
            class_diversity["overall"] = float(np.mean(diversity))
        for category, quality in {"overall": report["overall"], **report.get("per_class", {})}.items():
            score_rows.append({
                "framing": framing, "conditional": conditional, "seed": seed, "step": step,
                "method": method, "horizontal_flip_prob": flip_prob, "truncation": latent_limit,
                "category": category,
                "n_real": quality["n_real"], "n_generated": quality["n_generated"],
                "fid": quality["frechet_inception_distance"],
                "kid": quality["kernel_inception_distance_mean"], "is": quality["inception_score_mean"],
                "within_class_distance": class_diversity.get(category),
                "class_hit_rate": report.get("class_hit_rate", {}).get(
                    category, report.get("classifier_on_generated", {}).get("balanced_accuracy")
                    if category == "overall" else None
                ),
            })
        save_samples(generator, conditional, classes, device, folder / "final_samples",
                     truncation=latent_limit)
        (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{folder.name}: {report['overall']}", flush=True)

    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    for (seed, step, framing), variants in flip_comparisons.items():
        if 0.0 not in variants or len(variants) != 2:
            continue
        real, baseline = variants[0.0]
        (probability,) = variants.keys() - {0.0}
        _, flipped = variants[probability]
        panels = [
            [(115, "Real validation", real[val_labels == c][:1]),
             (240, "Translation baseline", baseline[c][:4]),
             (660, f"Translation + horizontal flip ({probability:.0%})", flipped[c][:4])]
            for c in range(len(classes))
        ]
        plot_grid(output / f"horizontal_flip_seed{seed}_step{step}.png",
                  f"{framing} | Matched noise | Seed {seed} | {step:,} updates", classes, panels)


    for key, basic_rows in plain_comparisons.items():
        seed, step, framing = key
        for probability, (real, current_rows) in flip_comparisons[key].items():
            if not probability:
                continue
            panels = [
                [(115, "Real validation", real[val_labels == c][:1]),
                 (240, "Basic AC-GAN", basic_rows[c][:4]),
                 (660, f"AC-GAN + translation + CBN + EMA + flip ({probability:.0%})", current_rows[c][:4])]
                for c in range(len(classes))
            ]
            plot_grid(output / f"basic_vs_current_seed{seed}_step{step}.png",
                      f"{framing} | Matched noise | Seed {seed} | {step:,} updates", classes, panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--inception-weights", type=Path)
    parser.add_argument("--overall-noise-seed", type=int, default=1234)
    parser.add_argument("--class-noise-seed", type=int, default=NOISE_SEED)
    parser.add_argument("--truncation", type=float, default=0.0,
                        help="Latent limit for conditional checkpoints; 0 uses standard Gaussian noise")
    args = parser.parse_args()
    if not math.isfinite(args.truncation) or args.truncation < 0:
        parser.error("--truncation must be finite and nonnegative")
    with threadpool_limits(limits=4):
        evaluate(args.checkpoints, args.output, args.data_dir, args.inception_weights,
                 args.overall_noise_seed, args.class_noise_seed, args.truncation)


if __name__ == "__main__":
    main()

