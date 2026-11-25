# 📦 WaferLO 发布与打包指南

本页说明如何检查源码、构建 wheel 安装包，以及在 Linux 服务器上安装发布版本。

## ✅ 发布前检查

在仓库根目录执行：

```bash
python -m pytest -q
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

第一条命令运行自动化测试；第二条命令将项目打包为 Python wheel。构建结果位于 `dist/`，例如：

```text
dist/waferlo-8.3.22-py3-none-any.whl
```

## 🧰 安装 wheel

普通 Python 环境：

```bash
python -m pip install dist/waferlo-8.3.22-py3-none-any.whl
```

服务器已有 CUDA PyTorch 时：

```bash
python -m pip install dist/waferlo-8.3.22-py3-none-any.whl --no-deps
```

安装后验证：

```bash
waferlo --help
python -m pip show waferlo
```

## 🖥️ 服务器复现环境

若需要复现服务器的固定依赖，先安装锁定文件，再安装项目本身：

```bash
python -m pip install -r requirements-server.txt
python -m pip install -e . --no-deps
```

The server lockfile already declares the PyTorch CUDA 12.1 package index, so no extra index argument is required. This is necessary for `torch==2.1.1+cu121`, which is not hosted on the default PyPI mirror.

## 🚀 发布检查清单

- 确认 `README.md`、`README.zh-CN.md` 和 `CITATION.cff` 中的仓库地址正确。
- 确认 wheel 中包含模型配置、数据集 YAML 和 `ultralytics/waferlo/cli.py`。
- 发布前记录版本号、Python 版本、PyTorch/CUDA 版本和主要验证结果。
