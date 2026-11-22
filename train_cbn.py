"""Train or sample the 64-pixel SYSU-Shape AC-GAN experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset

from acgan import ImageDataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "sysu-shape-dataset"
IMAGE_SIZE = 64
BATCH_SIZE = 32
EMA_DECAY = 0.999


class SquareResizeDataset(ImageDataset):
    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        pixels = torch.from_numpy(np.array(rgb)).permute(2, 0, 1)
        return pixels.float() / 127.5 - 1, label


def load_split(data_dir):
    manifest = json.loads((ROOT / "sysu_split.json").read_text(encoding="utf-8"))
    return {**manifest, **{
        key: [(data_dir / path, label) for path, label in manifest[key]]
        for key in ("train", "validation")
    }}


def dataset_tensors(rows):
    images, labels = zip(*SquareResizeDataset(rows, IMAGE_SIZE))
    return torch.stack(images), torch.tensor(labels)


def pixels(images):
    return images.add(1).mul(127.5).round().clamp(0, 255).byte()


def dcgan():
    generator = nn.Sequential(
        nn.ConvTranspose2d(100, 512, 4, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh(),
    )
    discriminator = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
        nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
        nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
        nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        nn.Conv2d(512, 1, 4, bias=False), nn.Flatten(),
    )
    for layer in [*generator, *discriminator]:
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(layer.weight, 0, 0.02)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight, 1, 0.02)
            nn.init.zeros_(layer.bias)
    return generator, discriminator


def translate(images, max_shift, rng):
    batch, _, height, width = images.shape
    device = images.device
    offsets = torch.randint(-max_shift, max_shift + 1, (2, batch, 1, 1),
                            device=device, generator=rng)
    rows = torch.arange(height, device=device)[None, :, None] + offsets[0] + max_shift
    cols = torch.arange(width, device=device)[None, None, :] + offsets[1] + max_shift
    padded = F.pad(images, (max_shift,) * 4).permute(0, 2, 3, 1)
    batch_index = torch.arange(batch, device=device)[:, None, None]
    return padded[batch_index, rows, cols].permute(0, 3, 1, 2).contiguous()


class ConditionalBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, original, n_classes):
        super().__init__(original.num_features, eps=original.eps, momentum=original.momentum)
        self.load_state_dict(original.state_dict())
        self.class_affine = nn.Parameter(torch.zeros(n_classes, 2, original.num_features))

    def forward(self, x, labels):
        gain, bias = self.class_affine[labels].unbind(dim=1)
        return super().forward(x) * (1 + gain[:, :, None, None]) + bias[:, :, None, None]


class ConditionalGenerator(nn.Module):
    def __init__(self, backbone, n_classes, conditional_batch_norm=False):
        super().__init__()
        self.n_classes = n_classes
        self.model = backbone
        self.conditional_batch_norm = conditional_batch_norm
        first = nn.ConvTranspose2d(100 + n_classes, 512, 4, bias=False)
        nn.init.normal_(first.weight, 0, 0.02)
        with torch.no_grad():
            first.weight[:100].copy_(backbone[0].weight)
        self.model[0] = first
        if conditional_batch_norm:
            for index in (1, 4, 7):  # 4x4, 8x8, and 16x16 feature maps.
                self.model[index] = ConditionalBatchNorm2d(self.model[index], n_classes)

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, self.n_classes).to(dtype=z.dtype)[:, :, None, None]
        x = torch.cat([z, one_hot], dim=1)
        if not self.conditional_batch_norm:
            return self.model(x)
        for layer in self.model:
            x = layer(x, labels) if isinstance(layer, ConditionalBatchNorm2d) else layer(x)
        return x


class AuxiliaryDiscriminator(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.features = backbone[:-2]
        self.source = backbone[-2:]
        self.classifier = nn.Linear(512 * 4 * 4, n_classes)

    def forward(self, rgb):
        h = self.features(rgb)
        return self.source(h), self.classifier(h.flatten(1))


def models(n_classes, conditional_batch_norm=False):
    generator, discriminator = dcgan()
    return (ConditionalGenerator(generator, n_classes, conditional_batch_norm),
            AuxiliaryDiscriminator(discriminator, n_classes))


@torch.inference_mode()
def sample(generator, labels, device, shared_noise=False, noise_seed=1234):
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(noise_seed)
    count = len(labels) // generator.n_classes if shared_noise else len(labels)
    z = torch.randn(count, 100, 1, 1, generator=rng, device=device)
    if shared_noise:
        z = z.repeat(generator.n_classes, 1, 1, 1)
    rgb = torch.cat([generator(x, y.to(device)).cpu() for x, y in zip(z.split(32), labels.split(32))])
    return pixels(rgb)


@torch.no_grad()
def calibrate_batch_norm(generator, label_probabilities, batch_size, batches):
    device = label_probabilities.device
    rng = torch.Generator(device=device).manual_seed(3210)
    norms = [layer for layer in generator.modules() if isinstance(layer, nn.BatchNorm2d)]
    momenta = [layer.momentum for layer in norms]
    for layer in norms:
        layer.reset_running_stats()
        layer.momentum = None
    generator.train()
    for _ in range(batches):
        z = torch.randn(batch_size, 100, 1, 1, device=device, generator=rng)
        labels = torch.multinomial(label_probabilities, batch_size, replacement=True, generator=rng)
        generator(z, labels)
    for layer, momentum in zip(norms, momenta):
        layer.momentum = momentum
    generator.eval()


def class_grid(sampled, reference, classes, output, step, name):
    canvas = Image.new("RGB", (1120, 630), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((10, 10), f"{name} | {step:,} updates | same 8 noise vectors per class",
              font=font, fill="black")
    draw.text((89, 40), "Real", font=font, fill="black")
    draw.text((205, 40), "Generated", font=font, fill="black")
    for label, name in enumerate(classes):
        y = 66 + label * 110
        draw.text((8, y + 40), name, font=font, fill="black")
        for column, rgb in enumerate([reference[label], *sampled[label]]):
            image = Image.fromarray(rgb.permute(1, 2, 0).numpy()).resize((100, 100))
            canvas.paste(image, (89 if column == 0 else 205 + 108 * (column - 1), y))
    canvas.save(output)


def train(split, config, device, output):
    classes = split["classes"]
    seed = config["seed"]
    steps = config["steps"]
    use_cbn = config["conditional_batch_norm"]
    torch.manual_seed(seed)

    train_images, train_labels = dataset_tensors(split["train"])
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
    val_images, val_labels = dataset_tensors(split["validation"])
    real = pixels(torch.stack([val_images[val_labels == label][0] for label in range(len(classes))]))
    probe = torch.arange(len(classes)).repeat_interleave(8)
    g, d = (model.to(device) for model in models(len(classes), use_cbn))
    ema = None
    if use_cbn:
        ema = AveragedModel(g, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
        ema.requires_grad_(False)
        ema.update_parameters(g)
        class_counts = train_labels.bincount(minlength=len(classes)).float().to(device)

    opt_g = torch.optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    aug_rng = torch.Generator(device=device).manual_seed(seed + 2)
    name = "AC-GAN + DiffAugment + CBN" if use_cbn else "Original AC-GAN (64px)"

    def d_input(images):
        return translate(images, 4, aug_rng) if use_cbn else images

    with (output / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "d_source", "d_class", "g_source", "g_class"])
        batches = iter(loader)
        for step in range(1, steps + 1):
            if (step - 1) % len(loader) == 0 and step > 1:
                batches = iter(loader)
            images, labels = (batch.to(device) for batch in next(batches))
            z = torch.randn(len(images), 100, 1, 1, generator=rng, device=device)
            fake = g(z, labels)

            opt_d.zero_grad(set_to_none=True)
            real_src, real_cls = d(d_input(images))
            fake_src, fake_cls = d(d_input(fake.detach()))
            d_src = F.softplus(-real_src).mean() + F.softplus(fake_src).mean()
            d_cls = F.cross_entropy(real_cls, labels) + F.cross_entropy(fake_cls, labels)
            (d_src + d_cls).backward()
            opt_d.step()

            d.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake_src, fake_cls = d(d_input(fake))
            g_src = F.softplus(-fake_src).mean()
            g_cls = F.cross_entropy(fake_cls, labels)
            (g_src + g_cls).backward()
            opt_g.step()
            if ema is not None:
                ema.update_parameters(g)
            d.requires_grad_(True)

            losses = torch.stack((d_src.detach(), d_cls.detach(), g_src.detach(), g_cls.detach())).cpu()
            if not torch.isfinite(losses).all():
                raise RuntimeError(f"Non-finite loss at update {step}")
            writer.writerow([step, *losses.tolist()])
            if step == 1 or step % 100 == 0:
                file.flush()
                print(f"Update {step}/{steps}: {losses.tolist()}", flush=True)
            if step % 2000 == 0 or step == steps:
                samples = sample(g, probe, device, shared_noise=True)
                class_grid(samples.unflatten(0, (len(classes), 8)), real, classes,
                           output / f"classes_{step:04d}.png", step, name)
                g.train()
                checkpoint = {"generator": g.state_dict(), "discriminator": d.state_dict(),
                              "step": step, "config": config, "classes": classes}
                if ema is not None:
                    calibrate_batch_norm(ema.module, class_counts, BATCH_SIZE, 32)
                    samples = sample(ema.module, probe, device, shared_noise=True)
                    class_grid(samples.unflatten(0, (len(classes), 8)), real, classes,
                               output / f"classes_ema_{step:04d}.png", step, f"{name} + EMA")
                    checkpoint["generator_ema"] = ema.module.state_dict()
                torch.save(checkpoint, output / f"model_{step:04d}.pt")
    return ema.module if ema is not None else g


def save_samples(generator, classes, device, output):
    rng = torch.Generator().manual_seed(1234)
    for mode, labels in (
        ("conditional", torch.arange(len(classes)).repeat_interleave(10)),
        ("random_class", torch.randint(len(classes), (10,), generator=rng)),
    ):
        folder = output / mode
        folder.mkdir()
        for index, (image, label) in enumerate(zip(sample(generator, labels, device), labels.tolist())):
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(folder / f"{index:03d}_{classes[label]}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "cbn"), default="cbn")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Generate samples from an existing 64px checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        classes = state["classes"]
        generator, _ = models(len(classes), state["config"].get("conditional_batch_norm", False))
        generator.load_state_dict(state.get("generator_ema", state["generator"]))
        generator.to(device)
        args.output.mkdir(parents=True)
    else:
        use_cbn = args.variant == "cbn"
        config = {"variant": args.variant, "steps": args.steps or (12000 if use_cbn else 6000),
                  "seed": args.seed, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
                  "conditional_batch_norm": use_cbn, "translation_max_px": 4 if use_cbn else 0}
        if use_cbn:
            config.update(ema_decay=EMA_DECAY, ema_bn_calibration_batches=32)
        split = load_split(args.data_dir)
        classes = split["classes"]
        args.output.mkdir(parents=True)
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (args.output / "split.json").write_text((ROOT / "sysu_split.json").read_text(encoding="utf-8"),
                                               encoding="utf-8")
        print(f"Device: {device}; model: {args.variant}", flush=True)
        generator = train(split, config, device, args.output)
    save_samples(generator, classes, device, args.output)


if __name__ == "__main__":
    main()
