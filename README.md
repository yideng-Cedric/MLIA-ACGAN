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

## Evaluation

The standard-noise comparison reports FID, KID, and Inception Score overall and per class, along with an independent classifier's class agreement. It compares 512 generated images with the 307 validation photos overall and uses 256 generated images per class. Inception weights download on first evaluation. Use `--inception-weights PATH` with a local copy and `--data-dir PATH` if your dataset is elsewhere.

In the matched seed-2026 run, the current model reduced KID from 0.1511 to 0.0745 and increased classifier agreement from 27.4% to 77.2% under standard sampling.

## Reproduce the earlier 64px full-image experiments

Use the environment from the [local setup](#local-windows-powershell). Run these commands from the repository root.

```bash
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

## Run the project

Both workflows use the dataset and split included in the repository. The current model saves its checkpoint to `runs/current/model_12000.pt`. The demo shows two held-out photos and two generated images per class. It saves the figure and individual 64×64 samples under `runs/demo/five_classes/`.

There are two ways to generate without specifying a class at the command line.

- **Random-class AC-GAN** reuses the trained conditional checkpoint. The script draws a label uniformly from the five classes for each image and supplies that label to the generator. Ten random draws may repeat or omit classes. Change `--seed` for new draws.
- **Unconditional GAN (DCGAN)** trains a separate model on the same images without class labels. At inference its generator receives only noise. Use this model for the unconditional comparison in the report.

### Local (Windows PowerShell)

1. From the repository root, activate your Python environment and install a PyTorch build for your hardware. Install the requirements once and set the CUDA variable in each training shell.

   ```powershell
   python -m pip install -r requirements.txt
   $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
   ```

2. Train the current model from scratch. If you already have its checkpoint, skip to step 3.

   ```powershell
   python train.py --framing crop64 --seed 2026 --steps 12000 --horizontal-flip-prob 0.25 --deterministic --output runs/current
   ```

3. Generate the five-class demo and open the figure. If the checkpoint is stored elsewhere, pass `--checkpoint "PATH_TO_MODEL.pt"`. Use `--data-dir PATH` if you moved the dataset.

   ```powershell
   python demo.py
   Invoke-Item 'runs/demo/five_classes/real_vs_generated.png'
   ```

   To show a single class, run `python demo.py --class-name bicycle --seed 17` and open `runs/demo/bicycle/real_vs_generated.png`. Valid classes are `airplane`, `bicycle`, `boat`, `car`, and `motorbike`. Change `--seed` to draw new samples.

4. For the matched baseline and metric comparison, run the commands below. Set `--truncation 1.5` on a separate evaluation to measure the demo's sampling setting.

   ```powershell
   python train.py --framing crop64 --seed 2026 --steps 12000 --plain-acgan --deterministic --output runs/baseline
   python evaluate.py runs/baseline/model_12000.pt runs/current/model_12000.pt --output runs/evaluation
   python evaluate.py runs/current/model_12000.pt --truncation 1.5 --output runs/evaluation_truncated
   ```

5. Generate ten images with random-class AC-GAN or the separately trained unconditional GAN. Both use standard Gaussian noise for this workflow. You can pass `--truncation 1.5` to random-class AC-GAN to use the live demo's noise distribution.

   ```powershell
   # Random-class AC-GAN
   python train.py --checkpoint runs/current/model_12000.pt --random-class --seed 2026 --output runs/demo/random_class

   # Unconditional GAN
   python train.py --framing crop64 --unconditional --seed 2026 --steps 12000 --deterministic --output runs/unconditional
   python train.py --checkpoint runs/unconditional/model_12000.pt --output runs/demo/unconditional
   ```

   Skip unconditional training if `runs/unconditional/model_12000.pt` already exists. For the conditional versus unconditional metrics, pass the conditional checkpoint first to `evaluate.py`.

### Rivanna (Bash)

1. Open [UVA Open OnDemand](https://ood.hpc.virginia.edu) and select **Clusters → HPC Shell Access**. Clone the repository inside the team folder. If `code` is already a clone, run `git -C code pull --ff-only` from the team folder.

   ```bash
   cd /standard/mlia/MLIA-ACGAN
   git clone https://github.com/yideng-Cedric/MLIA-ACGAN.git code
   ```

2. Request one GPU using the course allocation and wait for the compute shell to open.

   ```bash
   ijob -A mlia -p interactive --gres=gpu:1 -c 4 --mem 8192 -t 01:00:00
   ```

3. Load the UVA PyTorch container once in the GPU job shell.

   ```bash
   cd /standard/mlia/MLIA-ACGAN/code
   module load apptainer pytorch/2.9.0
   image="$CONTAINERDIR/pytorch-2.9.0.sif"
   export APPTAINERENV_CUBLAS_WORKSPACE_CONFIG=:4096:8
   ```

4. Train the current model for 12,000 updates. If the trained checkpoint already exists, skip to step 5.

   ```bash
   apptainer exec --nv "$image" python -u train.py \
     --framing crop64 --seed 2026 --steps 12000 \
     --horizontal-flip-prob 0.25 --deterministic \
     --output runs/current
   ```

5. Run the demo in the same shell. Open `/standard/mlia/MLIA-ACGAN/code/runs/demo/five_classes/real_vs_generated.png` in the Open OnDemand file browser.

   ```bash
   apptainer exec --nv "$image" python demo.py --checkpoint runs/current/model_12000.pt
   ```

6. Generate ten images per option in the same shell. The PNGs are saved in `runs/demo/random_class/` and `runs/demo/unconditional/`.

   ```bash
   # Random-class AC-GAN
   apptainer exec --nv "$image" python train.py --checkpoint runs/current/model_12000.pt \
     --random-class --seed 2026 --output runs/demo/random_class

   # Unconditional GAN
   apptainer exec --nv "$image" python -u train.py --framing crop64 --unconditional \
     --seed 2026 --steps 12000 --deterministic --output runs/unconditional
   apptainer exec --nv "$image" python train.py --checkpoint runs/unconditional/model_12000.pt \
     --output runs/demo/unconditional
   ```
