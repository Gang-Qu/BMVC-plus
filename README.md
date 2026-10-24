# BMVC+

Code for **BMVC+: An Enhanced Block Modulation Video Compression Codec for Large-scale Image Compression**, accepted by **IEEE Journal of Selected Topics in Signal Processing (JSTSP), 2025**.

BMVC+ studies block-modulation-based compression and reconstruction of large-scale images. This repository contains reconstruction networks, training and evaluation scripts, and example test images. The current evaluation entry point uses the Mamba-based `DualBMVC` network.

## Repository Structure

```text
BMVC-plus/
|-- model/
|   |-- Network_Mamba.py       # Mamba-based reconstruction network
|   `-- Network_SwinTrans.py   # Transformer-based network variant
|-- mamba_ssm/                # Bundled Mamba Python modules
|-- test_data/                # 16 example images: 00000.jpg to 00015.jpg
|-- opts.py                   # Paths and model/training parameters
|-- train.py                  # Training entry point (see note below)
|-- test.py                   # Evaluation entry point
`-- utils.py                  # Data loading, metrics, and checkpoint utilities
```

Generated logs, reconstruction results, and model checkpoints are not included in this release.

## Environment

The implementation targets **Linux with an NVIDIA GPU and CUDA**. The bundled Mamba code requires compiled CUDA extensions; copying the Python modules alone is not sufficient. CPU-only and macOS execution are not supported by the current CUDA-dependent implementation.

Create an isolated Python environment, then install a CUDA-enabled PyTorch build compatible with your GPU driver and CUDA toolkit. A starting setup is:

```bash
conda create -n bmvc-plus python=3.10 -y
conda activate bmvc-plus

# Install a CUDA-enabled PyTorch build appropriate for your system first.
python -m pip install numpy scipy opencv-python scikit-image scikit-learn
python -m pip install einops timm albumentations packaging ninja
python -m pip install causal-conv1d mamba-ssm --no-build-isolation
```

The installed Mamba package must provide `selective_scan_cuda` compatible with the bundled `mamba_ssm/` source and your PyTorch/CUDA versions. Additional Mamba utilities may require `transformers` or `triton`. These commands describe the dependencies inferred from the source, not a version-pinned, validated reproduction environment.

## Data Preparation

```bash
git clone https://github.com/Gang-Qu/BMVC-plus.git
cd BMVC-plus
```

The `test_data/` directory contains 16 JPEG images, numbered consecutively from `00000.jpg` to `00015.jpg`. Original image contents are preserved. Evaluation reads images directly from this directory and converts them to the luminance (Y) channel.

For training, prepare a separate directory containing readable image files and pass its path through `--train_data_path`. The loader applies random cropping, flipping, resizing, and luminance conversion. Training images must be large enough for the configured random crops. Keep data directories free of non-image files.

## Evaluation

Run the following command with a compatible checkpoint stored locally:

```bash
python test.py \
  --benchmark_path ./test_data \
  --test_weight_path /path/to/checkpoint.pth \
  --device cuda
```

The script reports per-image and mean **PSNR** and **SSIM** on the luminance channel. Side-by-side ground-truth and reconstructed images are saved under `test_results/<decoder_type>/MCA_100/test/`, and logs under `test_results/log/`.

The model configuration must match the checkpoint. Defaults in `opts.py` include an image size of `1080 x 1920`, block factors `cr=[4, 8]`, measurement size `[272, 240]`, three reconstruction stages, and 128 channels. The list-valued arguments in the current parser should be edited directly in `opts.py`; they are not configured to accept multiple command-line values. Replace the original machine-specific paths when using your own data.

## Training

**Current release note:** `train.py` imports `DualBMVC` from `model.network_4S_mamba`, which is not included in this repository. Before training, restore the intended module or, after confirming the intended architecture, change that import to the provided `model.Network_Mamba`. The command below assumes this import has been resolved.

```bash
python train.py \
  --train_data_path /path/to/training_images \
  --benchmark_path ./test_data \
  --pretrained_model_path /path/to/compatible_checkpoint.pth \
  --batch_size 1 \
  --lr 0.00005 \
  --epochs 2000 \
  --device cuda
```

For training from scratch, set `pretrained_model_path` to Python `None` in `opts.py`; the string `"None"` passed on the command line is not equivalent. Training creates `log/`, `results/`, and `weights/` locally. Training and evaluation have not been validated as part of this repository upload.

## Acknowledgments

This implementation uses PyTorch, Mamba, timm, and other open-source scientific computing libraries. Please retain and follow the applicable upstream license notices when reusing bundled components.
