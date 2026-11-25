<div align="center">

# WaferLO: Dual-Branch Convolution and Progressive Attention-Aware YOLO for Wafer Defect Detection

**A task-specific detector for wafer and industrial surface defects**

[![GitHub](https://img.shields.io/badge/GitHub-Sunburst--Li%2FWaferLO-181717?logo=github)](https://github.com/Sunburst-Li/WaferLO)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

</div>

## 🌐 Paper and project overview

This repository accompanies the paper **“WaferLO: Dual-Branch Convolution and Progressive Attention-Aware YOLO for Wafer Defect Detection.”** WaferLO is a task-specific detector built on a customized Ultralytics runtime. It supports, but is not limited to, wafer defects, DsPCBSD+, and GC10-DET. Any dataset following the documented YOLO detection format can be used.

**Authors:** Xuebo Li, Chong Li, Xinning Wang, Jingzhi Li, Wenqi Ren, and Jin Wu. Affiliations include Ocean University of China, University of Science and Technology Beijing, and Sun Yat-sen University Shenzhen Campus.

## 🛠️ Environment and installation

The validated server environment uses Python 3.8.19. Choose either `.venv` or Conda and install a matching PyTorch/torchvision build.

### Option A: Python `.venv`

```bash
python3.8 -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Option B: Conda

```bash
conda create -n waferlo python=3.8.19 -y
conda activate waferlo
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The package name is `waferlo`, while the Python import namespace remains `ultralytics`. Do not mix an official Ultralytics installation with this project in the same environment. Flash Attention is optional; PyTorch scaled-dot-product attention is used as a fallback.

After `python -m pip install -e .`, the `project.scripts` entry in `pyproject.toml` registers `waferlo` in the active environment. Verify it with:

```bash
waferlo --help
python -m pip show waferlo
```

If the launcher is not on `PATH`, use the equivalent module entry point:

```bash
python -m ultralytics.waferlo.cli --help
```

For an existing CUDA PyTorch environment:

```bash
python -m pip install -e . --no-deps
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__file__)"
```

For an existing CUDA PyTorch environment, preserve its packages with:

```bash
python -m pip install -e . --no-deps
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__file__)"
```

For the validated server package set, use:

```bash
python -m pip install -r requirements-server.txt
python -m pip install -e . --no-deps
```

The `requirements-server.txt` file already includes the official PyTorch CUDA 12.1 index, so it can install `torch==2.1.1+cu121` without an extra download argument.

The same server versions are also available through the `server` optional dependency group:

```bash
python -m pip install -e ".[server]" --extra-index-url https://download.pytorch.org/whl/cu121
```

If CUDA PyTorch is already installed, activate the environment using its actual environment type (`conda activate <environment-name>` for Conda or `source /path/to/venv/bin/activate` for a virtual environment), then use `--no-deps` to preserve the existing CUDA stack.

```bash
python -m pip install -e . --no-deps
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__file__)"
```

## 🧩 Model and dataset configuration

All supported datasets use the same model definition:

```text
ultralytics/cfg/models/WaferLO/WaferLO.yaml
```

Training uses `data=` or `--data` to select a dataset YAML. The model YAML describes the network; the dataset YAML defines the class count, class order, and train/validation/test paths.

| Dataset | YAML | Classes |
|---|---|---:|
| Wafer defects | `ultralytics/cfg/datasets/wafer.yaml` | 7 |
| DsPCBSD+ | `ultralytics/cfg/datasets/DsPCBSD+.yaml` | 9 |
| GC10-DET | `ultralytics/cfg/datasets/GC10-DET.yaml` | 10 |

The three published dataset templates use portable `path` values and relative split paths. After cloning the repository, edit only `path`:

```text
dataset-root/
├── train/images/       train/labels/
├── valid/images/       valid/labels/
└── test/images/        test/labels/
```

Labels use normalized `class_id x_center y_center width height` rows with coordinates in `[0, 1]`. Class IDs must exactly match the selected YAML. Datasets are not bundled or redistributed with this repository.

## 🗂️ Project structure

See [docs/structure.md](docs/structure.md) for the complete GitHub repository map and the responsibility of each source, configuration, and documentation directory.

## 📚 Documentation navigation

| Document | Description |
|---|---|
| [Project structure](docs/structure.md) | Repository files and module responsibilities |
| [Environment](docs/environment.md) | `.venv`, Conda, and server setup |
| [Datasets](docs/dataset.md) | Dataset YAML and label format |
| [CLI/Python guide](docs/cli-python.md) | Command-line and Python API mapping |
| [Baselines](docs/baselines.md) | Related papers and public implementations |
| [Release guide](docs/release.md) | Tests, wheel building, and installation |

## 🚀 CLI training, validation and prediction

### Wafer dataset

```bash
waferlo train --model waferlo --data ultralytics/cfg/datasets/wafer.yaml --epochs 300 --batch 24 --imgsz 800 --device 0
```

### DsPCBSD+ dataset

```bash
waferlo train --model waferlo --data ultralytics/cfg/datasets/DsPCBSD+.yaml --epochs 300 --batch 16 --imgsz 512 --device 0 --project runs/dspcbsd --name waferlo
```

### GC10-DET dataset

```bash
waferlo train --model waferlo --data ultralytics/cfg/datasets/GC10-DET.yaml --epochs 300 --batch 16 --imgsz 640 --device 0 --project runs/gc10-det --name waferlo
```

Before a full run, validate paths and arguments without starting training:

```bash
waferlo train --model waferlo --data ultralytics/cfg/datasets/wafer.yaml --dry-run
```

If training options are omitted, the WaferLO defaults are used. Initialization weights are loaded only when `--pretrained checkpoint.pt` is explicitly provided. Resume an interrupted run from `last.pt` with `--resume`.

Validation and test evaluation are run separately:

```bash
waferlo val --model runs/waferlo/train/weights/best.pt --data ultralytics/cfg/datasets/wafer.yaml --split val --imgsz 800 --batch 16 --device 0
waferlo val --model runs/waferlo/train/weights/best.pt --data ultralytics/cfg/datasets/wafer.yaml --split test --imgsz 800 --batch 16 --device 0 --save-json
```

Prediction and export:

```bash
waferlo predict --model runs/waferlo/train/weights/best.pt --source dataset-root/test/images --imgsz 800 --device 0
waferlo export --model runs/waferlo/train/weights/best.pt --format onnx --imgsz 800
```


## 🐍 Python API

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/WaferLO/WaferLO.yaml")
model.train(
    data="ultralytics/cfg/datasets/wafer.yaml",
    epochs=300, batch=24, imgsz=800, device=0,
    optimizer="auto", patience=300, project="runs/waferlo", name="train",
)
metrics = model.val(data="ultralytics/cfg/datasets/wafer.yaml", split="test")
results = model.predict(source="/path/to/images", save=True)
model.export(format="onnx")
```

For DsPCBSD+ and GC10-DET:

```python
from ultralytics import YOLO

for data_yaml, project, imgsz in [
    ("ultralytics/cfg/datasets/DsPCBSD+.yaml", "runs/dspcbsd", 512),
    ("ultralytics/cfg/datasets/GC10-DET.yaml", "runs/gc10-det", 640),
]:
    model = YOLO("ultralytics/cfg/models/WaferLO/WaferLO.yaml")
    model.train(data=data_yaml, epochs=300, batch=16, imgsz=imgsz,
                device=0, optimizer="auto", patience=300,
                project=project, name="waferlo")
```


Validation, prediction, and export:

```python
from ultralytics import YOLO

model = YOLO("runs/waferlo/train/weights/best.pt")
metrics = model.val(
    data="ultralytics/cfg/datasets/wafer.yaml",
    split="test", imgsz=800, batch=16, device=0,
)
results = model.predict(
    source="dataset-root/test/images",
    imgsz=800, device=0, save=True,
)
onnx_path = model.export(format="onnx", imgsz=800)
print(metrics.results_dict, onnx_path)
```

## ⚡ Train, Val, Predict and Export

The three datasets use the same model YAML; only the `data` path changes. The standard `yolo` interface is also available:

```bash
yolo detect train model=ultralytics/cfg/models/WaferLO/WaferLO.yaml data=ultralytics/cfg/datasets/wafer.yaml epochs=300 batch=24 imgsz=800 device=0
yolo detect train model=ultralytics/cfg/models/WaferLO/WaferLO.yaml data=ultralytics/cfg/datasets/DsPCBSD+.yaml epochs=300 batch=16 imgsz=512 device=0
yolo detect train model=ultralytics/cfg/models/WaferLO/WaferLO.yaml data=ultralytics/cfg/datasets/GC10-DET.yaml epochs=300 batch=16 imgsz=640 device=0
yolo detect val model=runs/waferlo/train/weights/best.pt data=ultralytics/cfg/datasets/wafer.yaml split=test
yolo detect predict model=runs/waferlo/train/weights/best.pt source=/path/to/images save=True
yolo export model=runs/waferlo/train/weights/best.pt format=onnx imgsz=800
```

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/WaferLO/WaferLO.yaml")
model.train(data="ultralytics/cfg/datasets/wafer.yaml", epochs=300, batch=24, imgsz=800, device=0)
metrics = YOLO("runs/waferlo/train/weights/best.pt").val(data="ultralytics/cfg/datasets/wafer.yaml", split="test")
for result in YOLO("runs/waferlo/train/weights/best.pt").predict(source="dataset-root/test/images", save=True):
    print(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
YOLO("runs/waferlo/train/weights/best.pt").export(format="onnx", imgsz=800)
```

Replace the `data` path with `DsPCBSD+.yaml` or `GC10-DET.yaml` to run the same workflow on the other datasets.

## 🙏 Acknowledgements

We acknowledge the open-source communities behind RT-DETR, DINO, D-FINE, Wafer-YOLO, and Ultralytics. Their projects are cited as related work and public references in [docs/baselines.md](docs/baselines.md); their implementations and checkpoints are not copied into this repository.

## 📦 Release and packaging

See [docs/release.md](docs/release.md) for testing, wheel creation, server installation, and the GitHub release checklist.

For more details, see the [dataset guide](docs/dataset.md), [environment guide](docs/environment.md), and [CLI/Python guide](docs/cli-python.md).

## 📜 Citation

If you use WaferLO in your research, please cite this project and the associated paper. The project homepage is [Sunburst-Li/WaferLO](https://github.com/Sunburst-Li/WaferLO).

| Citation method | File or format |
|---|---|
| GitHub / software citation | [CITATION.cff](CITATION.cff) |
| BibTeX | BibTeX entry below |

```bibtex
@article{li2026waferlo,
  title   = {WaferLO: Dual-Branch Convolution and Progressive Attention-Aware YOLO for Wafer Defect Detection},
  author  = {Li, Xuebo and Li, Chong and Wang, Xinning and Li, Jingzhi and Ren, Wenqi and Wu, Jin},
  journal = {Manuscript},
  year    = {2026}
}
```

## ⚖️ License

This project is released under the [AGPL-3.0 license](LICENSE).
