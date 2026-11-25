<div align="center">

# WaferLO: Dual-Branch Convolution and Progressive Attention-Aware YOLO for Wafer Defect Detection

[![GitHub](https://img.shields.io/badge/GitHub-Sunburst--Li%2FWaferLO-181717?logo=github)](https://github.com/Sunburst-Li/WaferLO)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

</div>

## 🌐 论文与项目简介

本仓库对应论文 **《WaferLO: Dual-Branch Convolution and Progressive Attention-Aware YOLO for Wafer Defect Detection》**。支持但不限于 Wafer 晶圆缺陷、DsPCBSD+ 和 GC10-DET 三种数据集模式。凡是符合文档所述 YOLO 检测格式的数据集均可使用。WaferLO 代码库组织：仓库同时提供模型配置、数据集模板、命令行入口、Python API、服务器依赖锁定文件和引用元数据，便于检查并扩展到其他 YOLO 格式数据集。

**作者：** Xuebo Li、Chong Li、Xinning Wang、Jingzhi Li、Wenqi Ren、Jin Wu。单位：Ocean University of China（School of Engineering）、University of Science and Technology Beijing（School of Intelligent Science and Technology）、Sun Yat-sen University Shenzhen Campus（School of Cyber Science and Technology）。

仓库公开 README：提供环境、安装、数据配置、CLI、Python API、训练/验证/推理/导出和其他说明。

## 🛠️ 环境与安装

支持 Python `.venv` 和 Conda 两种独立环境。建议环境使用 Python 3.8.19，并安装 PyTorch/torchvision 版本。

### 方式 A：Python `.venv`

```bash
python3.8 -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 方式 B：Conda

```bash
conda create -n waferlo python=3.8.19 -y
conda activate waferlo
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

发行包名为 `waferlo`，Python 导入命名空间仍为 `ultralytics`。请不要在同一环境中混装官方 Ultralytics 包和本项目。Flash Attention 是可选项；没有安装时自动使用 PyTorch 的 scaled-dot-product attention。

执行 `python -m pip install -e .` 后，`pyproject.toml` 的 `project.scripts` 会在当前虚拟环境中注册 `waferlo` 命令：Windows 位于 `.venv\Scripts\`，Linux/macOS 位于 `.venv/bin/`。可以使用 `waferlo --help` 和 `python -m pip show waferlo` 验证。若命令未加入 PATH，可执行等价入口：`python -m ultralytics.waferlo.cli --help`。

已有 CUDA PyTorch 的可使用：

```bash
python -m pip install -e . --no-deps
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__file__)"
```

如需复现精确包版本，可使用仓库内的锁定文件：

```bash
python -m pip install -r requirements-server.txt
python -m pip install -e . --no-deps
```

`requirements-server.txt` 已内置 PyTorch CUDA 12.1 官方软件源，因此可以直接安装 `torch==2.1.1+cu121`，无需额外添加下载地址。

也可以直接使用 `pyproject.toml` 中集成的依赖组：

```bash
python -m pip install -e ".[server]" --extra-index-url https://download.pytorch.org/whl/cu121
```

如果已经安装 CUDA PyTorch，请先按实际环境类型激活它：Conda 环境使用 `conda activate <环境名>`，普通虚拟环境使用 `source /path/to/venv/bin/activate`。然后使用 `--no-deps` 保留已有 CUDA PyTorch：

```bash
python -m pip install -e . --no-deps
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__file__)"
```

## 🧩 模型与数据集配置

三个任务都使用同一个模型 YAML：

```text
ultralytics/cfg/models/WaferLO/WaferLO.yaml
```

训练时通过 `data=` 或 `--data` 指定数据 YAML。模型 YAML 只描述网络；数据 YAML 描述类别数、类别顺序和训练/验证/测试路径。

| 任务 | 数据 YAML | 类别数 |
| --- | --- | ---: |
| Wafer 晶圆缺陷 | `ultralytics/cfg/datasets/wafer.yaml` | 7 |
| DsPCBSD+ | `ultralytics/cfg/datasets/DsPCBSD+.yaml` | 9 |
| GC10-DET | `ultralytics/cfg/datasets/GC10-DET.yaml` | 10 |

三个模板都使用可迁移的 `path` 和相对 split 路径。复制仓库后只需修改 `path`：

```text
数据集根目录/
├── train/images/       train/labels/
├── valid/images/       valid/labels/
└── test/images/        test/labels/
```

标签格式为 `class_id x_center y_center width height`，坐标归一化到 `[0,1]`。类别 ID 必须与所选 YAML 完全一致。数据集不包含在仓库中，也不由仓库重新分发。

## 🗂️ 项目目录结构

完整的 GitHub 项目目录说明见 [docs/structure.md](docs/structure.md)，其中包含模型代码、数据集 YAML、CLI 入口、环境依赖、文档和发布文件的职责说明。

## 📚 文档导航

| 文档 | 内容 |
|---|---|
| [项目目录结构](docs/structure.md) | 仓库文件和模块职责 |
| [环境配置](docs/environment.md) | `.venv`、Conda 和服务器环境 |
| [数据集说明](docs/dataset.md) | YAML、类别和标签格式 |
| [CLI/Python 对照](docs/cli-python.md) | 命令行与 Python API |
| [公开模型](docs/baselines.md) | 相关论文和公开代码 |
| [发布指南](docs/release.md) | 测试、wheel 打包和安装 |

## 🚀 CLI 训练、验证与推理

安装后使用 `waferlo`，也可以使用等价的 `python -m ultralytics.waferlo.cli`。

### Wafer 数据集

```bash
waferlo train --model waferlo \
  --data ultralytics/cfg/datasets/wafer.yaml \
  --epochs 300 --batch 24 --imgsz 800 --device 0
```

### DsPCBSD+ 数据集

```bash
waferlo train --model waferlo \
  --data ultralytics/cfg/datasets/DsPCBSD+.yaml \
  --epochs 300 --batch 16 --imgsz 512 --device 0 \
  --project runs/dspcbsd --name waferlo
```

### GC10-DET 数据集

```bash
waferlo train --model waferlo \
  --data ultralytics/cfg/datasets/GC10-DET.yaml \
  --epochs 300 --batch 16 --imgsz 640 --device 0 \
  --project runs/gc10-det --name waferlo
```

正式训练前可以只检查路径和参数：

```bash
waferlo train --model waferlo --data ultralytics/cfg/datasets/wafer.yaml --dry-run
```

不指定训练参数将会使用默认训练参数。只有显式传入 `--pretrained checkpoint.pt` 才会加载初始化权重；从 `last.pt` 继续训练使用 `--resume`。

验证集和测试集分开执行：

```bash
waferlo val --model runs/waferlo/train/weights/best.pt \
  --data ultralytics/cfg/datasets/wafer.yaml --split val \
  --imgsz 800 --batch 16 --device 0

waferlo val --model runs/waferlo/train/weights/best.pt \
  --data ultralytics/cfg/datasets/wafer.yaml --split test \
  --imgsz 800 --batch 16 --device 0 --save-json
```

预测和导出：

```bash
waferlo predict --model runs/waferlo/train/weights/best.pt \
  --source 数据集根目录/test/images --imgsz 800 --device 0
waferlo export --model runs/waferlo/train/weights/best.pt --format onnx --imgsz 800
```

## 🐍 Python API

Python API 与 CLI 使用同一模型 YAML；只切换 `data`：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/WaferLO/WaferLO.yaml")
model.train(
    data="ultralytics/cfg/datasets/wafer.yaml",
    epochs=300, batch=24, imgsz=800, device=0,
    optimizer="auto", project="runs/waferlo", name="train",
)
```

DsPCBSD+ 和 GC10-DET：

```python
from ultralytics import YOLO

for data_yaml, project, imgsz in [
    ("ultralytics/cfg/datasets/DsPCBSD+.yaml", "runs/dspcbsd", 512),
    ("ultralytics/cfg/datasets/GC10-DET.yaml", "runs/gc10-det", 640),
]:
    model = YOLO("ultralytics/cfg/models/WaferLO/WaferLO.yaml")
    model.train(data=data_yaml, epochs=300, batch=16, imgsz=imgsz,
                device=0, optimizer="auto", project=project, name="waferlo")
```

验证、推理、导出：

```python
from ultralytics import YOLO

model = YOLO("runs/waferlo/train/weights/best.pt")
metrics = model.val(data="ultralytics/cfg/datasets/wafer.yaml", split="test", imgsz=800, device=0)
results = model.predict(source="数据集根目录/test/images", imgsz=800, device=0, save=True)
onnx_path = model.export(format="onnx", imgsz=800)
print(metrics.results_dict, onnx_path)
```

## ⚡ Train、Val、Predict 与 Export

三个数据集共用同一个模型 YAML，只替换 `data=`。

```bash
yolo detect train model=ultralytics/cfg/models/WaferLO/WaferLO.yaml data=ultralytics/cfg/datasets/WaferLO.yaml epochs=100 imgsz=640
yolo detect train model=ultralytics/cfg/models/WaferLO/WaferLO.yaml data=ultralytics/cfg/datasets/DsPCBSD+.yaml epochs=100 imgsz=640
yolo detect train model=ultralytics/cfg/models/WaferLO/WaferLO.yaml data=ultralytics/cfg/datasets/GC10-DET.yaml epochs=100 imgsz=640
yolo detect val model=runs/detect/train/weights/best.pt data=ultralytics/cfg/datasets/WaferLO.yaml
yolo detect predict model=runs/detect/train/weights/best.pt source=/path/to/images save=True
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/WaferLO/WaferLO.yaml")
model.train(data="ultralytics/cfg/datasets/WaferLO.yaml", epochs=100, imgsz=640)
metrics = YOLO("runs/detect/train/weights/best.pt").val(data="ultralytics/cfg/datasets/WaferLO.yaml")
for result in YOLO("runs/detect/train/weights/best.pt").predict(source="/path/to/images", save=True):
    print(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
YOLO("runs/detect/train/weights/best.pt").export(format="onnx")
```

将 `data` 替换为 `DsPCBSD+.yaml` 或 `GC10-DET.yaml` 即可运行另外两个数据集。

## 🙏 致谢

RT-DETR、DINO、D-FINE、Wafer-YOLO 以及 Ultralytics 是开源项目，由来自多所高校和企业的研究人员与工程师共同贡献。各项目的来源见 [docs/baselines.md](docs/baselines.md)。我们感谢所有实现其方法或添加新功能的贡献者，以及提供宝贵反馈的用户。

## 📦 发布与安装包

项目打包、wheel 安装和源码发布说明见 [docs/release.md](docs/release.md)。

更多说明见 [数据集文档](docs/dataset.md)、[环境配置](docs/environment.md) 和 [CLI/Python 对照](docs/cli-python.md)。

## 📜 引用

如果您在研究中使用 WaferLO，请引用本项目及对应论文。项目主页为 [Sunburst-Li/WaferLO](https://github.com/Sunburst-Li/WaferLO)。

| 引用方式 | 文件或格式 |
|---|---|
| GitHub / 软件引用 | [CITATION.cff](CITATION.cff) |
| BibTeX | 见下方 BibTeX 条目 |

```bibtex
@article{li2026waferlo,
  title   = {WaferLO: Dual-Branch Convolution and Progressive Attention-Aware YOLO for Wafer Defect Detection},
  author  = {Li, Xuebo and Li, Chong and Wang, Xinning and Li, Jingzhi and Ren, Wenqi and Wu, Jin},
  journal = {Manuscript},
  year    = {2026}
}
```

## ⚖️ 许可证

[本作品采用知识共享署名-非商业性使用 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc/4.0/)进行许可，仅限非商业用途。任何商业用途均需事先获得正式许可。
