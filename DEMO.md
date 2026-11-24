# Live demo

Start in the repository root with the Python dependencies installed and your project environment active. The demo needs the local SYSU-Shape images and a trained checkpoint. The [README](README.md#train-and-evaluate) has the training command.

## Show all five classes

```powershell
python demo.py
Invoke-Item 'runs/demo/five_classes/real_vs_generated.png'
```

The default checkpoint is `runs/current/model_12000.pt`. If you already have the trained checkpoint elsewhere, run `python demo.py --checkpoint "PATH_TO_MODEL.pt"`. Use `--data-dir PATH` if you moved the dataset.

Each row shows two held-out real photos on the left and two newly generated images on the right. The script loads the saved EMA generator and draws one independent noise vector per image. Save the figure beforehand as a presentation backup. The original 64×64 generated PNGs are in `runs/demo/five_classes/generated`.

## Show a class chosen during the demo

```powershell
python demo.py --class-name bicycle --seed 17
Invoke-Item 'runs/demo/bicycle/real_vs_generated.png'
```

Valid class names are `airplane`, `bicycle`, `boat`, `car`, and `motorbike`. Change `--seed` to draw new samples. The output folder is named after the selected class.

## Optional noise-only experiment

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python train.py --framing crop64 --unconditional --seed 2026 --steps 12000 --deterministic --output runs/unconditional
python train.py --checkpoint runs/unconditional/model_12000.pt --output runs/demo/unconditional
```

Train the noise-only GAN ahead of time. The second command generates ten images from its saved checkpoint. For the conditional versus unconditional metric comparison, pass both checkpoints to `evaluate.py` with the conditional checkpoint first.
