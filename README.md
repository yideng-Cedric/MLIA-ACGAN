# SYSU-Shape AC-GAN

## Current method

The model generates 64×64 RGB images of airplanes, bicycles, boats, cars, and motorbikes. It keeps the AC-GAN objective. The generator receives 100 noise values and a class label. The discriminator scores both realism and class, and the generator learns to satisfy both tasks.

| Part | Implementation |
| --- | --- |
| Data | Crop around the annotated objects with a small margin and resize to 64×64. Use the supplied split of 1,234 training and 307 validation photos. |
| Generator | Apply conditional BatchNorm at 4×4, 8×8, and 16×16 so the class influences the generator beyond its input. |
| Training augmentation | Apply differentiable translations of up to 4 pixels and horizontal flips with 25% probability to real and generated images seen by the discriminator. |
| Checkpoint | Use the exponential moving average of generator weights for sampling, with decay 0.999. |
| Inference | The live demo samples Gaussian noise truncated at ±1.5. This improved class recognition in our experiments and reduced within-class variation. |

The matched plain AC-GAN baseline uses the same crop, data split, and training budget. It uses standard BatchNorm and trains without augmentation or EMA. Both models train with standard Gaussian noise.

## Train and evaluate

Run from the repository root with the project Python environment active. Install a PyTorch build suitable for your hardware, then run `python -m pip install -r requirements.txt`.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python train.py --framing crop64 --seed 2026 --steps 12000 --horizontal-flip-prob 0.25 --deterministic --output runs/current
python train.py --framing crop64 --seed 2026 --steps 12000 --plain-acgan --deterministic --output runs/baseline
python evaluate.py runs/baseline/model_12000.pt runs/current/model_12000.pt --output runs/evaluation
```

The standard-noise comparison reports FID, KID, and Inception Score overall and per class, along with an independent classifier's class agreement. It compares 512 generated images with the 307 validation photos overall and uses 256 generated images per class. Inception weights download on first evaluation. Use `--inception-weights PATH` with a local copy and `--data-dir PATH` if your dataset is elsewhere. To evaluate the demo's sampling setting, run `python evaluate.py runs/current/model_12000.pt --truncation 1.5 --output runs/evaluation_truncated`.

In the matched seed-2026 run, the current model reduced KID from 0.1511 to 0.0745 and increased classifier agreement from 27.4% to 77.2% under standard sampling.

## Live demo

```powershell
python demo.py
```

The figure shows two held-out photos and two generated images per class. See the [live demo runbook](DEMO.md) for setup, changing the class or noise seed, and the optional noise-only experiment. Model checkpoints and generated outputs live under the Git-ignored `runs/` directory. Train locally or pass an existing checkpoint with `--checkpoint PATH`.

## Reproduce the earlier 64px full-image experiments

Install a PyTorch build for your hardware, then install the project requirements. Commands below run from the repository root.

```bash
python -m pip install -r requirements.txt
python train_cbn.py --variant original --seed 42 --output runs/original_acgan/plain_64/seed42
python train_cbn.py --variant cbn --seed 42 --output runs/cbn/seed42
```

The original 64px run uses 6,000 updates. The CBN run uses 12,000 updates and EMA decay 0.999. Repeat with `--seed 123` for the second seed. Choose a fresh output folder for every training run. Both load the same relative `sysu_split.json` with 1,234 training images and 307 validation images. The default dataset is `sysu-shape-dataset/`. Use `--data-dir PATH` if you keep the dataset elsewhere.

```bash
python evaluate_cbn.py runs/original_acgan/plain_64/seed42/model_6000.pt runs/cbn/seed42/model_12000.pt --output runs/evaluation_seed42
python train_cbn.py --checkpoint runs/cbn/seed42/model_12000.pt --output runs/generated_seed42
```

Evaluation saves FID, KID, Inception Score, classifier agreement, and a grid of real validation images alongside matched-noise generated images. It downloads the standard torch-fidelity Inception weights on first use. For an offline run, pass a local copy with `--inception-weights PATH`. The sampling command uses EMA when the checkpoint contains it. `random_class` samples labels at random from the five classes.

## Original 128px implementation

```bash
python main.py --architecture paper --image-size 128 --epochs 100 --output runs/original_acgan/paper_128
```

The original model reconstructs the paper appendix architecture for this dataset, with the non-saturating generator loss by default. Use `--g-loss minimax` for the main-text objective. [AC-GAN paper](https://proceedings.mlr.press/v70/odena17a.html)

## Saved results

- [Original 128px sample grid](results/original/paper128.png)
- [Matched-noise comparison with real images, seed 42](results/seed42/comparison.png) and [metrics](results/seed42/metrics.json)
- [Matched-noise comparison with real images, seed 123](results/seed123/comparison.png) and [metrics](results/seed123/metrics.json)
- [Two-seed metric comparison plot](results/metrics_comparison.png), generated with `python3 plot_results.py`

The comparison grids show the archived 6,000-step 64px baseline and the 12,000-step CBN run. They summarize the two complete training recipes. Full local checkpoints, samples, and logs are in the Git-ignored `runs/` directory.
