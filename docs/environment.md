# 环境配置

## Supported baseline

### Python `.venv`

```bash
python3.8 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Conda

```bash
conda create -n waferlo python=3.8.19 -y
conda activate waferlo
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The supported range is Python 3.8–3.12 with a matching PyTorch/torchvision installation. New environments are recommended to use Python 3.8.19; the validated server environment uses Python 3.8.19. CPU execution is supported for model construction, validation and small smoke tests. CUDA execution is recommended for training at 512–800 pixel inputs.

Check the active environment before a run:

```bash
python --version
python -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(ultralytics.__file__)"
```

The last path must point to this checkout. If an official `ultralytics` package appears first, create a clean virtual environment or reinstall this project with `python -m pip install -e .`.

## CPU

Install the CPU build of PyTorch from the official PyTorch selector, then install WaferLO:

```bash
python -m pip install -e "."
```

Use `--device cpu` for model checks and small experiments.

## CUDA

Install the PyTorch wheel matching the host CUDA driver, then install WaferLO without replacing that PyTorch build:

```bash
python -m pip install -e . --no-deps
```

Verify the device before training:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

The project does not pin one CUDA version because the correct wheel depends on the host driver. Record the output of the commands above with each experiment.

## Optional export dependencies

ONNX export and runtime comparison require additional packages. Install the relevant extras for the target deployment, then compare exported outputs with the PyTorch checkpoint on representative images:

```bash
python -m pip install onnx onnxruntime
waferlo export --model runs/waferlo/train/weights/best.pt --format onnx --imgsz 800
```

Flash Attention, TensorRT, OpenVINO and other accelerators are optional deployment components; they are not required to train the WaferLO model.
