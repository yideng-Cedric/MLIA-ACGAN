"""Data -> AC-GAN training -> save model -> generate -> optional evaluation."""

# Run from this project directory in PowerShell; use a new output folder per run.
# Paper architecture, 128x128:
# python.exe main.py --architecture paper --image-size 128 --epochs 100 --output runs/paper_128
# Paper architecture adapted to 64x64:
# python.exe main.py --architecture paper --image-size 64 --epochs 100 --output runs/paper_64
# Packaged TorchGAN model, 64x64:
# python.exe main.py --architecture torchgan --image-size 64 --epochs 100 --output runs/torchgan_64

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch

from acgan import CLASSES, build_models, load_data, train, generate, evaluate

ROOT = Path(__file__).resolve().parent

CONFIG = dict(architecture="paper", image_size=128, latent_dim=100, channels=64,
              batch_size=64, epochs=100, lr=0.0002, seed=42, g_loss="non_saturating")
DATA_DIR = ROOT / "sysu-shape-dataset"
EVALUATE = False  
SAMPLES_PER_CLASS = 10
RANDOM_SAMPLES = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--architecture", choices=["baseline", "torchgan", "paper"], default=CONFIG["architecture"])
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128], default=CONFIG["image_size"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--g-loss", choices=["non_saturating", "minimax"], default=CONFIG["g_loss"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--max-steps", type=int, help="A short pipeline check only")
    args = parser.parse_args()
    config = {**CONFIG, "epochs": args.epochs, "architecture": args.architecture,
              "image_size": args.image_size, "batch_size": args.batch_size, "g_loss": args.g_loss}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    args.output = args.output or ROOT / "runs" / f"{args.architecture}_{args.image_size}"
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate":
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        config = checkpoint["config"]

    # 1. Load the five-class dataset and save the split.
    if args.mode == "train" or EVALUATE:
        loader, validation = load_data(DATA_DIR, config["batch_size"], config["image_size"], config["seed"])
        split = {"classes": CLASSES, "train": loader.dataset.samples, "validation": validation}
        (args.output / "split.json").write_text(json.dumps(split, indent=2))

    # 2. Build the generator and two-head discriminator.
    generator, discriminator = build_models(config, device)
    (args.output / "architecture.txt").write_text(f"{generator}\n\n{discriminator}")
    print(f"Device: {device}; output: {args.output}", flush=True)

    # 3. Train, or load a generator for independent inference.
    if args.mode == "train":
        config["torch_version"] = str(torch.__version__)
        config["torchgan_version"] = version("torchgan") if config["architecture"] == "torchgan" else None
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        train(generator, discriminator, loader, device, args.output, config, args.max_steps)
    else:
        generator.load_state_dict(checkpoint["generator"])

    # 4. Specified-class and random-class sampling use the same generator.
    samples = generate(generator, device, args.output, SAMPLES_PER_CLASS, RANDOM_SAMPLES)
    print("Generation finished.", flush=True)

    # 5. Optional overall + five per-class FID, KID and IS.
    if EVALUATE:
        evaluate(validation, samples, config["image_size"], device, args.output)


if __name__ == "__main__":
    main()
