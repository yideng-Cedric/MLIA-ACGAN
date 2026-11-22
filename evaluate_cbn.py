"""Evaluate one or two SYSU-Shape 64px AC-GAN checkpoints and compare samples."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.nn.functional import normalize
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics

from train_cbn import DATA_DIR, dataset_tensors, load_split, models, pixels, sample


@torch.inference_mode()
def features(extractor, images, device):
    rows = [extractor(batch.to(device)) for batch in images.split(32)]
    return [torch.cat([row[index].cpu() for row in rows]) for index in range(len(rows[0]))]


def agreement(prediction, expected, classes):
    return {"n": len(expected), "correct": int((prediction == expected).sum()),
            "balanced_accuracy": float(balanced_accuracy_score(expected, prediction)),
            "confusion_matrix": confusion_matrix(expected, prediction, labels=range(len(classes))).tolist()}


def distribution_metrics(fake, logits, validation, real_stats):
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_features_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric

    stats = fid_statistics_to_metric(fid_features_to_statistics(fake), real_stats, verbose=False)
    stats.update(kid_features_to_metric(fake, validation, kid_subset_size=64,
                                        kid_subsets=100, rng_seed=42, verbose=False))
    stats.update(isc_features_to_metric(logits, splits=10, rng_seed=42))
    return stats


def within_class_distance(encodings, labels, n_classes):
    encodings = normalize(encodings)
    distances = []
    for label in range(n_classes):
        members = encodings[labels == label]
        n = len(members)
        similarity = (members.sum(dim=0).square().sum() - n) / (n * (n - 1))
        distances.append(float(1 - similarity))
    return distances


def comparison_grid(groups, headings, classes, output):
    size, stride = 80, 84
    starts = [94 + 184 + 526 * i for i in range(len(groups) - 1)]
    canvas = Image.new("RGB", (starts[-1] + 528, 505), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    title = ImageFont.load_default(size=20)
    draw.text((12, 10), "SYSU-Shape | 64x64 class-conditional generation", font=title, fill="black")
    draw.text((12, 36), "Real images are held out. Generated columns use the same noise across rows and models.",
              font=font, fill="black")
    for label, category in enumerate(classes):
        draw.text((12, 84 + stride * label + 28), category, font=font, fill="black")
    for x, heading, samples in zip((94, *starts), headings, groups):
        draw.text((x, 63), heading, font=font, fill="black")
        for label, row in enumerate(samples):
            y = 84 + stride * label
            for column, rgb in enumerate(row[:6]):
                image = Image.fromarray(rgb.permute(1, 2, 0).numpy()).resize((size, size))
                canvas.paste(image, (x + stride * column, y))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inception-weights", type=Path, help="Optional local torch-fidelity Inception weights")
    args = parser.parse_args()
    if len(args.checkpoints) > 2:
        parser.error("provide one or two checkpoints")

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(args.data_dir)
    classes = split["classes"]
    n_classes = len(classes)
    train_images, train_labels = dataset_tensors(split["train"])
    val_images, val_labels = dataset_tensors(split["validation"])
    val_images = pixels(val_images)
    weights_path = str(args.inception_weights) if args.inception_weights is not None else None
    extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"],
                                            feature_extractor_weights_path=weights_path).to(device).eval()
    train_features, _ = features(extractor, pixels(train_images), device)
    val_features, _ = features(extractor, val_images, device)
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(normalize(train_features).numpy(), train_labels.numpy())
    val_prediction = classifier.predict(normalize(val_features).numpy())
    real_stats = fid_features_to_statistics(val_features)

    indices = torch.linspace(0, len(val_labels) - 1, 256).long()
    metric_labels = val_labels[indices]
    metric_labels = metric_labels[torch.randperm(len(metric_labels), generator=torch.Generator().manual_seed(42))]
    samples_per_class = 256
    target = torch.arange(n_classes).repeat_interleave(samples_per_class)
    report = {
        "classes": classes, "fid_kid_samples": {"real": len(val_labels), "generated": len(metric_labels)},
        "class_control_noise_seed": 56017, "class_control_samples_per_class": samples_per_class,
        "real_validation": agreement(val_prediction, val_labels.numpy(), classes), "models": [],
    }
    groups = [torch.stack([val_images[val_labels == label][:2] for label in range(n_classes)])]
    headings = ["Real validation"]
    for path in args.checkpoints:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        config = checkpoint["config"]
        if checkpoint["classes"] != classes or config["image_size"] != 64:
            parser.error("checkpoints must use the five SYSU-Shape classes at 64x64")
        use_cbn = config.get("conditional_batch_norm", False)
        generator, _ = models(n_classes, use_cbn)
        generator.load_state_dict(checkpoint.get("generator_ema", checkpoint["generator"]))
        generator.to(device)

        metric_images = sample(generator, metric_labels, device)
        fake_features, logits = features(extractor, metric_images, device)
        quality = distribution_metrics(fake_features, logits, val_features, real_stats)
        fresh_images = sample(generator, target, device, shared_noise=True, noise_seed=56017)
        fresh_features, _ = features(extractor, fresh_images, device)
        prediction = classifier.predict(normalize(fresh_features).numpy())
        result = {
            "method": "diffaugment_cbn_ema" if use_cbn else "original_acgan",
            "seed": config["seed"], "step": checkpoint["step"],
            "metrics": quality,
            "independent_generated": agreement(prediction, target.numpy(), classes),
            "within_class_inception_cosine_distance": within_class_distance(fresh_features, target, n_classes),
        }
        report["models"].append(result)
        name = "+ DiffAugment + CBN + EMA" if use_cbn else "Original AC-GAN"
        headings.append(f"{name} | {checkpoint['step']:,} updates")
        groups.append(fresh_images.unflatten(0, (n_classes, samples_per_class)))
        print(name, json.dumps(result), flush=True)

    args.output.mkdir(parents=True)
    comparison_grid(groups, headings, classes, args.output / "comparison.png")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
