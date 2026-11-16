"""Basic AC-GAN functions: https://proceedings.mlr.press/v70/odena17a.html"""

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

CLASSES = ["airplane", "bicycle", "boat", "car", "motorbike"]


class ImageDataset(Dataset):
    def __init__(self, samples, image_size=64, for_metrics=False):
        self.samples = samples
        self.image_size = image_size
        self.for_metrics = for_metrics

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = ImageOps.pad(image.convert("RGB"), (self.image_size, self.image_size),
                                 method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return image if self.for_metrics else (image.float() / 127.5 - 1, label)


def load_data(data_dir, batch_size=64, image_size=64, seed=42):
    train, validation = [], []
    rng = random.Random(seed)
    for label, name in enumerate(CLASSES):
        paths = sorted(p for p in (Path(data_dir) / name / "images").iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(paths)
        n_val = round(len(paths) * 0.2)
        validation.extend((str(p), label) for p in paths[:n_val])
        train.extend((str(p), label) for p in paths[n_val:])
        print(f"{name}: train={len(paths) - n_val}, validation={n_val}")
    loader = DataLoader(ImageDataset(train, image_size), batch_size=batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    return loader, validation


class Generator(nn.Module):
    def __init__(self, latent_dim=100, image_size=64, channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(len(CLASSES), latent_dim)
        levels = image_size.bit_length() - 3
        self.start_channels = channels * 2 ** (levels - 1)
        self.project = nn.Linear(2 * latent_dim, self.start_channels * 4 * 4)
        width = self.start_channels
        layers = [nn.BatchNorm2d(width), nn.ReLU(True)]
        for _ in range(levels - 1):
            layers += [nn.ConvTranspose2d(width, width // 2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(width // 2), nn.ReLU(True)]
            width //= 2
        self.network = nn.Sequential(*layers, nn.ConvTranspose2d(width, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z, labels):
        x = torch.cat([z, self.embedding(labels)], dim=1)
        return self.network(self.project(x).view(-1, self.start_channels, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, image_size=64, channels=64):
        super().__init__()
        layers, width = [], 3
        for level in range(image_size.bit_length() - 3):
            next_width = channels * 2 ** level
            layers += [nn.Conv2d(width, next_width, 4, 2, 1),
                       nn.LeakyReLU(0.2, True), nn.Dropout2d(0.3)]
            width = next_width
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.source = nn.Linear(width * 4 * 4, 1)
        self.classifier = nn.Linear(width * 4 * 4, len(CLASSES))

    def forward(self, images):
        features = self.features(images)
        return self.source(features).flatten(), self.classifier(features)


def paper_init(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, 0, 0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)


class PaperGenerator(nn.Module):
    """Appendix Table 2 at 32px; Table 1 at 128px; explicit 64px adaptation."""
    def __init__(self, latent_dim=100, image_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        widths = [384, 192, 96, 3] if image_size == 32 else [768, 384, 256, 192, 3]
        self.start_size = image_size // 2 ** (len(widths) - 1)
        self.start_channels = widths[0]
        self.project = nn.Linear(latent_dim + len(CLASSES), widths[0] * self.start_size ** 2)
        layers = []
        for index, (in_ch, out_ch) in enumerate(zip(widths, widths[1:])):
            # padding=2/output_padding=1 implements a 2x spatial increase.
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 5, 2, 2, output_padding=1))
            if index < len(widths) - 2:
                layers += [nn.BatchNorm2d(out_ch), nn.ReLU(True)]
            else:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.apply(paper_init)

    def forward(self, z, labels):
        labels = F.one_hot(labels, len(CLASSES)).to(z.dtype)
        x = F.relu(self.project(torch.cat([z, labels], dim=1)))
        x = x.view(-1, self.start_channels, self.start_size, self.start_size)
        return self.network(x)


class PaperDiscriminator(nn.Module):
    """Six 3x3 convolutions, alternating strides, BN, and elementwise dropout."""
    def __init__(self, image_size=128):
        super().__init__()
        layers, width = [], 3
        for index, (out_ch, stride) in enumerate(zip([16, 32, 64, 128, 256, 512], [2, 1, 2, 1, 2, 1])):
            layers.append(nn.Conv2d(width, out_ch, 3, stride, 1))
            if index > 0:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.LeakyReLU(0.2, True), nn.Dropout(0.5)]
            width = out_ch
        self.features = nn.Sequential(*layers, nn.Flatten())
        features = 512 * (image_size // 8) ** 2
        self.source = nn.Linear(features, 1)
        self.classifier = nn.Linear(features, len(CLASSES))
        self.apply(paper_init)

    def forward(self, images):
        features = self.features(images)
        return self.source(features).flatten(), self.classifier(features)


class TorchGANDiscriminator(nn.Module):
    def __init__(self, image_size=64, channels=64):
        super().__init__()
        from torchgan.models import ACGANDiscriminator
        self.model = ACGANDiscriminator(num_classes=len(CLASSES), in_size=image_size,
            in_channels=3, step_channels=channels, last_nonlinearity=nn.Identity())

    def forward(self, images):
        return self.model(images, mode="both")


def build_models(config, device):
    architecture = config.get("architecture", "baseline")
    size, latent, channels = config["image_size"], config["latent_dim"], config["channels"]
    if architecture == "paper":
        generator, discriminator = PaperGenerator(latent, size), PaperDiscriminator(size)
    elif architecture == "torchgan":
        from torchgan.models import ACGANGenerator
        generator = ACGANGenerator(num_classes=len(CLASSES), encoding_dims=latent,
            out_size=size, out_channels=3, step_channels=channels)
        generator.latent_dim = latent
        discriminator = TorchGANDiscriminator(size, channels)
    else:
        generator, discriminator = Generator(latent, size, channels), Discriminator(size, channels)
    return generator.to(device), discriminator.to(device)


def train(generator, discriminator, loader, device, output, config, max_steps=None):
    opt_g = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    generator.train()
    discriminator.train()
    rng = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    step = 0
    with (output / "training.csv").open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "step", "d_source", "d_class", "g_source", "g_class"])
        for epoch in range(config["epochs"]):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                z = torch.randn(len(images), config["latent_dim"], generator=rng, device=device)
                fake_labels = torch.randint(len(CLASSES), (len(images),), generator=rng, device=device)
                fake = generator(z, fake_labels)

                # D: real/fake discrimination + classes of real AND generated images.
                opt_d.zero_grad()
                real_source, real_class = discriminator(images)
                fake_source, fake_class = discriminator(fake.detach())
                d_source = (bce(real_source, torch.ones_like(real_source)) +
                            bce(fake_source, torch.zeros_like(fake_source)))
                d_class = ce(real_class, labels) + ce(fake_class, fake_labels)
                (d_source + d_class).backward()
                opt_d.step()

                # G: fool D's source head and match the requested class.
                discriminator.requires_grad_(False)
                opt_g.zero_grad()
                fake_source, fake_class = discriminator(fake)
                g_source = (-bce(fake_source, torch.zeros_like(fake_source))
                            if config.get("g_loss") == "minimax"
                            else bce(fake_source, torch.ones_like(fake_source)))
                g_class = ce(fake_class, fake_labels)
                (g_source + g_class).backward()
                opt_g.step()
                discriminator.requires_grad_(True)

                step += 1
                writer.writerow([epoch + 1, step, d_source.item(), d_class.item(),
                                 g_source.item(), g_class.item()])
                if step == 1 or step % 20 == 0:
                    print(f"Epoch {epoch + 1}, step {step}: "
                          f"D={(d_source + d_class).item():.4f}, "
                          f"G={(g_source + g_class).item():.4f}", flush=True)
                if max_steps and step >= max_steps:
                    break
            log.flush()
            torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                        "config": config, "class_names": CLASSES, "epoch": epoch + 1,
                        "step": step}, output / "model.pt")
            if (epoch + 1) % 25 == 0:
                torch.save({"generator": generator.state_dict(), "config": config,
                            "epoch": epoch + 1, "step": step}, output / f"generator_{epoch + 1:03d}.pt")
            if max_steps and step >= max_steps:
                break


@torch.no_grad()
def generate(generator, device, output, per_class=10, random_count=10, seed=1234):
    generator.eval()
    rng = torch.Generator().manual_seed(seed)
    samples = {"conditional": [], "random_class": []}
    jobs = [("conditional", torch.full((per_class,), label)) for label in range(len(CLASSES))]
    jobs.append(("random_class", torch.randint(len(CLASSES), (random_count,), generator=rng)))
    for mode, labels in jobs:
        for start in range(0, len(labels), 64):
            batch_labels = labels[start:start + 64]
            z = torch.randn(len(batch_labels), generator.latent_dim, generator=rng).to(device)
            images = generator(z, batch_labels.to(device)).add(1).mul(127.5).round().clamp(0, 255).byte().cpu()
            for index, (image, label) in enumerate(zip(images, batch_labels.tolist()), start):
                folder = output / mode / (CLASSES[label] if mode == "conditional" else "")
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{index:05d}.png"
                Image.fromarray(image.permute(1, 2, 0).numpy()).save(path)
                samples[mode].append((str(path), label))
    (output / "samples.json").write_text(json.dumps({"seed": seed, "samples": samples,
        "note": "random_class uses the conditional model with uniformly random labels."}, indent=2))
    return samples


def evaluate(validation, samples, image_size, device, output):
    import torch_fidelity

    groups = [("conditional_overall", samples["conditional"], validation)]
    groups += [(name, [p for p in samples["conditional"] if p[1] == label],
                [p for p in validation if p[1] == label]) for label, name in enumerate(CLASSES)]
    groups.append(("random_class_overall", samples["random_class"], validation))
    results = []
    for name, fake, real in groups:
        scores = torch_fidelity.calculate_metrics(
            input1=ImageDataset(fake, image_size, for_metrics=True),
            input2=ImageDataset(real, image_size, for_metrics=True),
            cuda=device.type == "cuda", isc=True, fid=True, kid=True, verbose=False,
            batch_size=32, kid_subset_size=min(100, len(fake), len(real)),
            kid_subsets=100, isc_splits=min(10, max(1, len(fake) // 10)), rng_seed=1234)
        results.append({"group": name, "n_real": len(real), "n_generated": len(fake), **scores})
        print(name, scores, flush=True)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
