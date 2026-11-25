"""Train or sample the 64×64 SYSU-Shape GANs."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from train_cbn import (BATCH_SIZE, DATA_DIR, EMA_DECAY, calibrate_batch_norm, dcgan,
                       load_split, models, pixels, translate)


FORMATS = ("full64", "crop64")
IMAGE_SIZE = 64
SIZE_Z = 100


def crop_box(path, width, height):
    annotation = path.parent.parent / "annotations" / f"{path.stem}.mat"
    objects = np.atleast_1d(loadmat(annotation, squeeze_me=True, struct_as_record=False)["object"])
    x, y = [], []
    for obj in objects:
        ymin, _, xmin, _ = map(int, obj.box_coord)
        px, py = np.asarray(obj.obj_contour, dtype=np.int64)
        x.extend(px + xmin)
        y.extend(py + ymin)
    left, right = max(0, min(x) - 1), min(width, max(x))
    top, bottom = max(0, min(y) - 1), min(height, max(y))
    pad_x, pad_y = 0.10 * (right - left), 0.10 * (bottom - top)
    left, right = max(0, math.floor(left - pad_x)), min(width, math.ceil(right + pad_x))
    top, bottom = max(0, math.floor(top - pad_y)), min(height, math.ceil(bottom + pad_y))
    box_w, box_h = right - left, bottom - top
    side = max(box_w, box_h)
    crop_w, crop_h = min(width, side), min(height, side)
    x0 = min(width - crop_w, max(0, round((left + right - crop_w) / 2)))
    y0 = min(height - crop_h, max(0, round((top + bottom - crop_h) / 2)))
    return x0, y0, x0 + crop_w, y0 + crop_h


def image_tensor(path, framing):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if framing == "crop64":
            image = image.crop(crop_box(path, *image.size))
        rgb = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
    return torch.from_numpy(rgb).permute(2, 0, 1)


def make_models(n_classes, conditional, use_cbn):
    if conditional:
        return models(n_classes, conditional_batch_norm=use_cbn)
    return dcgan()


@torch.inference_mode()
def sample(generator, labels, device, shared_noise=False, noise_seed=1234, truncation=0.0):
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(noise_seed)
    count = len(labels) // generator.n_classes if shared_noise else len(labels)
    noise = torch.randn(count, SIZE_Z, 1, 1, device=device, generator=rng)
    if truncation:
        scale = math.sqrt(2)
        mass = math.erf(truncation / scale)
        noise = scale * torch.erfinv(mass * torch.erf(noise / scale))
    if shared_noise:
        noise = noise.repeat(generator.n_classes, 1, 1, 1)
    rgb = torch.cat([generator(z, y.to(device)).cpu()
                     for z, y in zip(noise.split(BATCH_SIZE), labels.split(BATCH_SIZE))])
    return pixels(rgb)


@torch.inference_mode()
def sample_noise(generator, count, device, noise_seed):
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(count, SIZE_Z, 1, 1, device=device, generator=rng)
    return pixels(torch.cat([generator(z).cpu() for z in noise.split(BATCH_SIZE)]))


@torch.no_grad()
def calibrate_unconditional(generator, device):
    rng = torch.Generator(device=device).manual_seed(3210)
    norms = [layer for layer in generator.modules() if isinstance(layer, nn.BatchNorm2d)]
    momenta = [layer.momentum for layer in norms]
    generator.train()
    for layer in norms:
        layer.reset_running_stats()
        layer.momentum = None
    for _ in range(32):
        generator(torch.randn(BATCH_SIZE, SIZE_Z, 1, 1, device=device, generator=rng))
    for layer, momentum in zip(norms, momenta):
        layer.momentum = momentum
    generator.eval()


def load_generation_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    conditional = config.get("conditional", True)
    use_cbn = config.get("conditional_batch_norm", False)
    generator, _ = make_models(len(checkpoint["classes"]), conditional, use_cbn)
    generator.load_state_dict(checkpoint.get("generator_ema", checkpoint["generator"]))
    return generator.to(device).eval(), config, checkpoint["classes"], checkpoint["step"]


def save_samples(generator, conditional, classes, device, output, class_name=None, truncation=0.0,
                 *, random_class_seed=None):
    output.mkdir(parents=True, exist_ok=True)
    if random_class_seed is not None:
        rng = torch.Generator().manual_seed(random_class_seed)
        labels = torch.randint(len(classes), (10,), generator=rng)
        noise_seed = (rng.initial_seed() + 1) % 2**64
        images = sample(generator, labels, device, noise_seed=noise_seed, truncation=truncation)
        names = [f"random_{i + 1:02}" for i in range(len(labels))]
    elif conditional:
        selected = classes if class_name is None else [class_name]
        labels = torch.tensor([classes.index(name) for name in selected]).repeat_interleave(2)
        images = sample(generator, labels, device, shared_noise=class_name is None,
                        noise_seed=56017, truncation=truncation)
        names = [f"{name}_{i + 1}" for name in selected for i in range(2)]
    else:
        images = sample_noise(generator, 10, device, noise_seed=56017)
        names = [f"noise_{i + 1:02}" for i in range(10)]
    for name, rgb in zip(names, images):
        Image.fromarray(rgb.permute(1, 2, 0).numpy()).save(output / f"{name}.png")


def train(split, framing, conditional, seed, steps, device, output, horizontal_flip_prob=0.0, plain_acgan=False):
    classes = split["classes"]
    torch.manual_seed(seed)
    images = torch.stack([image_tensor(path, framing) for path, _ in split["train"]])
    if conditional:
        labels = torch.tensor([label for _, label in split["train"]])
        rows = list(zip(images, labels))
        class_counts = labels.bincount(minlength=len(classes)).float().to(device)
    else:
        rows = [(image,) for image in images]
    loader = DataLoader(rows, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    use_cbn = conditional and not plain_acgan
    translation_max_px = 0 if plain_acgan else 4
    g, d = [module.to(device) for module in make_models(len(classes), conditional, use_cbn)]
    ema = None
    if not plain_acgan:
        ema = AveragedModel(g, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
        ema.requires_grad_(False)
        ema.update_parameters(g)
    opt_g = torch.optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
    noise_rng = torch.Generator(device=device).manual_seed(seed + 1)
    aug_rng = torch.Generator(device=device).manual_seed(seed + 2)
    flip_rng = torch.Generator(device=device).manual_seed(seed + 3)
    config = dict(framing=framing, conditional=conditional, seed=seed, steps=steps,
                  size_width_height=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
                  translation_max_px=translation_max_px, horizontal_flip_prob=horizontal_flip_prob,
                  conditional_batch_norm=use_cbn, plain_acgan=plain_acgan, adam_lr=0.0002,
                  deterministic=torch.are_deterministic_algorithms_enabled())
    if ema is not None:
        config.update(ema_decay=EMA_DECAY, ema_bn_batches=32)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def discriminate(rgb):
        if horizontal_flip_prob:
            flip = torch.rand(len(rgb), 1, 1, 1, device=device, generator=flip_rng) < horizontal_flip_prob
            rgb = torch.where(flip, rgb.flip(-1), rgb)
        if translation_max_px:
            rgb = translate(rgb, translation_max_px, aug_rng)
        result = d(rgb)
        return result if conditional else (result, None)

    batches = iter(loader)
    with (output / "training.csv").open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["step", "d_loss", "g_loss"])
        for step in range(1, steps + 1):
            if step > 1 and (step - 1) % len(loader) == 0:
                batches = iter(loader)
            rgb, *labels = [tensor.to(device) for tensor in next(batches)]
            rgb = rgb.float().div_(127.5).sub_(1)
            z = torch.randn(len(rgb), SIZE_Z, 1, 1, device=device, generator=noise_rng)
            fake = g(z, *labels)

            opt_d.zero_grad(set_to_none=True)
            real_src, real_cls = discriminate(rgb)
            fake_src, fake_cls = discriminate(fake.detach())
            d_loss = F.softplus(-real_src).mean() + F.softplus(fake_src).mean()
            if conditional:
                (target,) = labels
                d_loss = d_loss + (F.cross_entropy(real_cls, target) + F.cross_entropy(fake_cls, target))
            d_loss.backward()
            opt_d.step()

            d.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake_src, fake_cls = discriminate(fake)
            g_loss = F.softplus(-fake_src).mean()
            if conditional:
                g_loss = g_loss + F.cross_entropy(fake_cls, target)
            g_loss.backward()
            opt_g.step()
            if ema is not None:
                ema.update_parameters(g)
            d.requires_grad_(True)
            writer.writerow([step, d_loss.item(), g_loss.item()])
            if step == 1 or step % 250 == 0:
                log.flush()
                print(f"{framing} conditional={conditional} seed={seed} step {step}/{steps}", flush=True)

            if step % 4000 == 0 or step == steps:
                checkpoint = dict(config=config, classes=classes, step=step, generator=g.state_dict(),
                                  discriminator=d.state_dict())
                sample_model = g
                if ema is not None:
                    sample_model = ema.module
                    if conditional:
                        calibrate_batch_norm(sample_model, class_counts, BATCH_SIZE, 32)
                    else:
                        calibrate_unconditional(sample_model, device)
                    checkpoint["generator_ema"] = sample_model.state_dict()
                save_samples(sample_model, conditional, classes, device, output / f"samples_{step}")
                g.train()
                torch.save(checkpoint, output / f"model_{step}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framing", choices=FORMATS, default="crop64")
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--plain-acgan", action="store_true",
                        help="Use standard BatchNorm and train without augmentation or EMA")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PyTorch kernels")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42, help="Seed for training or random-class sampling")
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.0,
                        help="Per-image probability of flipping horizontally before D")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path)
    class_choice = parser.add_mutually_exclusive_group()
    class_choice.add_argument("--class-name", help="Select a category when sampling a conditional checkpoint")
    class_choice.add_argument("--random-class", action="store_true",
                              help="Use a randomly drawn category for each of ten conditional samples")
    parser.add_argument("--truncation", type=float, default=0.0,
                        help="Latent limit for conditional sampling; 0 uses standard Gaussian noise")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.horizontal_flip_prob <= 1:
        parser.error("--horizontal-flip-prob must be between 0 and 1")
    if args.plain_acgan and (args.unconditional or args.horizontal_flip_prob):
        parser.error("--plain-acgan requires conditional training without horizontal flips")
    if not math.isfinite(args.truncation) or args.truncation < 0 or (args.truncation and not args.checkpoint):
        parser.error("--truncation requires a nonnegative finite value and a checkpoint")
    if (args.class_name or args.random_class) and not args.checkpoint:
        parser.error("--class-name and --random-class require --checkpoint")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        generator, config, classes, _ = load_generation_checkpoint(args.checkpoint, device)
        conditional = config.get("conditional", True)
        if args.class_name and (not conditional or args.class_name not in classes):
            parser.error("--class-name requires a conditional checkpoint and a valid SYSU class")
        if (args.truncation or args.random_class) and not conditional:
            parser.error("--truncation and --random-class require a conditional checkpoint")
        save_samples(generator, conditional, classes, device, args.output, args.class_name, args.truncation,
                     random_class_seed=args.seed if args.random_class else None)
    else:
        train(load_split(args.data_dir), args.framing, not args.unconditional,
              args.seed, args.steps, device, args.output, args.horizontal_flip_prob, args.plain_acgan)


if __name__ == "__main__":
    main()

