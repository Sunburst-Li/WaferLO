# Dataset configuration

WaferLO uses standard Ultralytics YOLO detection labels. Each image has a matching `.txt` file with normalized `class x_center y_center width height` rows. The dataset YAML selects the root and split directories; the model YAML remains `ultralytics/cfg/models/WaferLO/WaferLO.yaml` for every dataset.

## Included configurations

| Dataset | YAML | Classes |
|---|---|---:|
| Wafer defects | `ultralytics/cfg/datasets/WaferLO.yaml` | 7 |
| DsPCBSD+ | `ultralytics/cfg/datasets/DsPCBSD+.yaml` | 9 |
| GC10-DET | `ultralytics/cfg/datasets/GC10-DET.yaml` | 10 |

Edit `path` in the selected YAML or pass an absolute dataset root. Keep `train`, `val`, and optional `test` paths relative to `path`.

## Recommended layout

```text
datasets/<name>/
├── train/images/   train/labels/
├── valid/images/   valid/labels/
└── test/images/    test/labels/
```

Class IDs must be contiguous from zero and agree with `nc` and `names`. The class count is read from the dataset YAML at runtime; do not create a second model YAML for each dataset.
