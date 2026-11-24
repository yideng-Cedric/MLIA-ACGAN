# SYSU-Shape AC-GAN

This project generates five categories of SYSU-Shape images from a noise vector and a class label. The 64px model adds DiffAugment translation, conditional BatchNorm at 4×4, 8×8, and 16×16, and an exponential moving average (EMA) of generator weights. The original 128px paper-style implementation is in `main.py` and `acgan.py`.

## Reproduce the 64px experiments

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
