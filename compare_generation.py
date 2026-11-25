"""Export and evaluate specified-class, random-class, and noise-only generation."""
import argparse
from pathlib import Path
import json

import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from torch.nn.functional import normalize
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

from evaluate import N_CLASS, N_OVERALL, agreement, features, metrics, plot_grid
from train import DATA_DIR, image_tensor, load_generation_checkpoint, load_split, sample, sample_noise


NOISE_SEED = 1234
LABEL_SEED = 2026
EXAMPLE_SEED = 56017


def save_examples(output, images, names):
    output.mkdir(parents=True, exist_ok=True)
    for rgb, name in zip(images, names):
        Image.fromarray(rgb.permute(1, 2, 0).numpy()).save(output / name)


def class_counts(labels, classes):
    counts = torch.bincount(torch.as_tensor(labels), minlength=len(classes))
    return dict(zip(classes, counts.tolist()))


def compare(conditional_path, unconditional_path, output, data_dir, weights_path):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(data_dir)
    classes = split["classes"]
    conditional, conditional_config, conditional_classes, conditional_step = load_generation_checkpoint(
        conditional_path, device)
    noise_only, noise_config, noise_classes, noise_step = load_generation_checkpoint(
        unconditional_path, device)
    if (not conditional_config["conditional"] or noise_config["conditional"]
            or conditional_classes != classes or noise_classes != classes):
        raise ValueError("Pass a conditional and a noise-only checkpoint for the supplied split.")
    framing = conditional_config["framing"]
    if noise_config["framing"] != framing:
        raise ValueError("Both checkpoints must use the same image preprocessing.")

    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048", "logits_unbiased"],
        feature_extractor_weights_path=weights_path,
    ).to(device).eval()
    train_paths, train_labels = zip(*split["train"])
    val_paths, val_labels = zip(*split["validation"])
    val_labels = torch.tensor(val_labels)
    train_features, _ = features(extractor, torch.stack([image_tensor(p, framing) for p in train_paths]), device)
    real_features, _ = features(extractor, torch.stack([image_tensor(p, framing) for p in val_paths]), device)
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(normalize(train_features).numpy(), train_labels)
    val_prediction = classifier.predict(normalize(real_features).numpy())

    label_indices = (torch.arange(N_OVERALL) * len(val_labels)) // N_OVERALL
    shuffle = torch.randperm(N_OVERALL, generator=torch.Generator().manual_seed(42))
    specified = val_labels[label_indices][shuffle]
    random_labels = torch.randint(len(classes), (N_OVERALL,), generator=torch.Generator().manual_seed(LABEL_SEED))
    case_labels = torch.arange(len(classes)).repeat_interleave(N_CLASS)
    class_images = sample(conditional, case_labels, device, shared_noise=True, noise_seed=EXAMPLE_SEED)
    class_features, class_logits = features(extractor, class_images, device)
    class_prediction = classifier.predict(normalize(class_features).numpy())
    per_class = {}
    for c, name in enumerate(classes):
        selected = case_labels == c
        per_class[name] = metrics(real_features[val_labels == c], class_features[selected], class_logits[selected], 32)
        per_class[name]["class_hit_rate"] = float((class_prediction[selected.numpy()] == c).mean())

    report = {
        "settings": {
            "framing": framing,
            "n_real_validation": len(val_labels),
            "n_generated_per_mode": N_OVERALL,
            "n_generated_per_conditional_class": N_CLASS,
            "sampling": "standard_gaussian",
            "noise_seed": NOISE_SEED,
            "random_class_seed": LABEL_SEED,
            "conditional_label_mix": "validation_frequencies",
            "random_label_mix": "uniform",
            "class_agreement": "Balanced accuracy against the generator's input labels. Noise-only samples have no target label.",
        },
        "classifier_on_validation": agreement(val_prediction, val_labels.numpy(), classes),
        "runs": {
            "conditional": {"checkpoint": conditional_path.as_posix(),
                            "training_seed": conditional_config["seed"], "training_steps": conditional_step},
            "unconditional_gan": {"checkpoint": unconditional_path.as_posix(),
                                  "training_seed": noise_config["seed"], "training_steps": noise_step},
        },
        "modes": {},
    }
    sets = (
        ("conditional", sample(conditional, specified, device, noise_seed=NOISE_SEED), specified),
        ("random_class", sample(conditional, random_labels, device, noise_seed=NOISE_SEED), random_labels),
        ("unconditional_gan", sample_noise(noise_only, N_OVERALL, device, NOISE_SEED), None),
    )
    for name, images, labels in sets:
        encoding, logits = features(extractor, images, device)
        prediction = classifier.predict(normalize(encoding).numpy())
        counts = class_counts(prediction, classes)
        row = {
            **metrics(real_features, encoding, logits, 64),
            "class_agreement": None if labels is None else agreement(prediction, labels.numpy(), classes),
            "input_class_counts": None if labels is None else class_counts(labels, classes),
            "predicted_class_counts": counts,
            "predicted_class_coverage": sum(count > 0 for count in counts.values()),
        }
        report["modes"][name] = row
        print(f"{name}: FID {row['frechet_inception_distance']:.2f}, "
              f"KID {row['kernel_inception_distance_mean']:.4f}", flush=True)

    report["modes"]["conditional"]["per_class"] = per_class
    report["modes"]["conditional"]["class_agreement"] = agreement(
        class_prediction, case_labels.numpy(), classes)
    named_examples = [f"{name}_{i + 1:02}.png" for name in classes for i in range(2)]
    specified_examples = class_images.unflatten(0, (len(classes), N_CLASS))[:, :2].flatten(0, 1)
    examples = (specified_examples, sets[1][1][:10], sets[2][1][:10])
    save_examples(output / "conditional", examples[0], named_examples)
    for name, images in zip(("random_class", "unconditional_gan"), examples[1:]):
        save_examples(output / name, images, [f"{i + 1:02}.png" for i in range(10)])
    report["settings"]["example_sampling"] = {
        "conditional": {"noise_seed": EXAMPLE_SEED, "samples_per_class": 2},
        "random_class": {"input_labels": [classes[c] for c in random_labels[:10].tolist()]},
    }
    rows = [[(115, heading, images)] for heading, images in zip(
        ("Two samples per class", "AC-GAN with internally sampled labels", "GAN trained without labels"), examples)]
    plot_grid(output / "comparison.png", "SYSU-Shape | Three generation modes",
              ["Conditional", "Random class", "Noise only"], rows)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional", type=Path, default=Path("runs/current/model_12000.pt"))
    parser.add_argument("--unconditional", type=Path, default=Path("runs/unconditional/model_12000.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/generation"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--inception-weights", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        compare(args.conditional, args.unconditional, args.output, args.data_dir, args.inception_weights)
