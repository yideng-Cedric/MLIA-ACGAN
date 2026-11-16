# AC-GAN

- `acgan.py`: models, data loading, training, generation, and optional evaluation.
- `main.py`: settings and the complete pipeline. Training commands are also at the top.
- Data: `sysu-shape-dataset/<class>/images`; outputs: `runs/`.

## Models

| `--architecture` | Model |
|---|---|
| `paper` (default) | Original appendix architecture at 32/128px; 64px is an adaptation. |
| `torchgan` | Packaged TorchGAN 0.1.0 AC-GAN. |
| `baseline` | Previous custom model, retained for old checkpoints. |

The paper model uses five classes. Unspecified reshape/padding details are reconstructed;
BN uses PyTorch defaults and activation noise is omitted. Default G loss is
non-saturating; `--g-loss minimax` selects the main-text objective.

## Run

Use the local `.venv` below; on another machine install `requirements.txt`
in a suitable PyTorch environment. Run from the project directory in PowerShell.

```powershell
# Paper architecture, 128x128
python main.py --architecture paper --image-size 128 --output runs/paper_128
# Packaged model, 64x64
python main.py --architecture torchgan --image-size 64 --output runs/torchgan_64
# Generate from a saved checkpoint
python main.py --mode generate --checkpoint runs/paper_128/model.pt --output runs/samples
```

Edit `CONFIG` and `DATA_DIR` in `main.py`. Defaults: batch 64, 100 epochs,
Adam 0.0002, seed 42; fixed stratified 80/20 split. Use a new output folder per run.
Each run saves settings, split, loss CSV, checkpoints, and generated images.
`random_class` samples random labels using the same conditional model.

For optional FID/KID/IS, install `torch-fidelity`, enable `EVALUATE`, and increase
sample counts. Architecture reconstruction does not imply reproduction of paper scores.

## References

[Paper](https://proceedings.mlr.press/v70/odena17a/odena17a.pdf) ·
[Architecture appendix](https://proceedings.mlr.press/v70/odena17a/odena17a-supp.pdf) ·
[TorchGAN](https://github.com/torchgan/torchgan/blob/master/torchgan/models/acgan.py) ·
[PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py) ·
[Keras-GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py)
