# CLI 与 Python 对照

Both interfaces use the same model configuration and the same Ultralytics backend. The only dataset-specific input is the selected dataset YAML.

| Operation | CLI | Python |
| --- | --- | --- |
| Train | `waferlo train --model waferlo --data DATA.yaml` | `YOLO(MODEL).train(data=DATA.yaml, ...)` |
| Validate | `waferlo val --model WEIGHTS --data DATA.yaml --split test` | `YOLO(WEIGHTS).val(data=DATA.yaml, split="test")` |
| Predict | `waferlo predict --model WEIGHTS --source IMAGES` | `YOLO(WEIGHTS).predict(source=IMAGES)` |
| Export | `waferlo export --model WEIGHTS --format onnx` | `YOLO(WEIGHTS).export(format="onnx")` |

Model names are flexible: `waferlo`, `waferlo.yaml`, and `WaferLO.yaml` resolve to the bundled `WaferLO.yaml`; an absolute or relative path to another `.yaml`, `.yml`, or `.pt` file is passed through unchanged.

For a fair comparison, keep `data`, `epochs`, `batch`, `imgsz`, `device`, `workers`, `seed`, `optimizer`, augmentation settings, `project`, and `name` identical. The `waferlo` entry point prints its resolved plan; save that output with the run.

Common settings are exposed directly, while less-common Ultralytics settings can be passed without changing this wrapper through repeatable overrides:

```bash
waferlo train --model waferlo --data DATA.yaml --override amp=True --override cache=False --override lr0=0.001
waferlo val --model WEIGHTS --data DATA.yaml --split test --override plots=True
waferlo predict --model WEIGHTS --source IMAGES --override conf=0.25 --override iou=0.7 --override half=True
```

The underlying defaults remain in `ultralytics/cfg/default.yaml`; use `--override KEY=VALUE` for run-specific values rather than editing the shared defaults file.

The same `WaferLO.yaml` is used for all three tasks:

```python
from ultralytics import YOLO

model_yaml = "ultralytics/cfg/models/WaferLO/WaferLO.yaml"
data_yaml = "ultralytics/cfg/datasets/GC10-DET.yaml"  # or wafer.yaml / DsPCBSD+.yaml
model = YOLO(model_yaml)
model.train(data=data_yaml, epochs=300, imgsz=640, batch=16, device=0)
```

Dataset class counts come from `data_yaml`; the same network is rebuilt with the selected number of detection classes at runtime. Do not edit the model YAML for a dataset class-count change.
