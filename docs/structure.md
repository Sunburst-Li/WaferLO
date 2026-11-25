# 🗂️ 项目目录结构

WaferLO follows a research-oriented layout while keeping the Ultralytics package namespace. The structure below describes the complete source repository prepared for GitHub publication.

```text
WaferLO/
├── LICENSE                         # AGPL-3.0 license text
├── README.md / README.zh-CN.md     # Project overview and usage
├── CITATION.cff                    # Citation metadata
├── pyproject.toml                  # Package metadata and dependencies
├── requirements-server.txt         # Validated server-version lockfile
├── docs/                           # Dataset, environment and release docs
│   ├── baselines.md                # Related papers and public code
│   ├── cli-python.md               # CLI/Python command mapping
│   ├── dataset.md                  # Dataset YAML and label format
│   ├── environment.md              # .venv, Conda and server setup
│   ├── release.md                  # Build and release workflow
│   └── structure.md                # This repository map
├── ultralytics/                    # Runtime and model implementation
│   ├── cfg/                        # Default, model and dataset YAML files
│   │   ├── datasets/               # WaferLO, DsPCBSD+ and GC10-DET YAMLs
│   │   └── models/WaferLO/         # The WaferLO model YAML
│   ├── nn/                         # Neural-network modules and custom layers
│   ├── engine/                     # Training, validation and prediction engine
│   ├── models/                     # Task model wrappers
│   ├── waferlo/                    # WaferLO CLI entry point
│   │   ├── __init__.py
│   │   └── cli.py
│   ├── data/                       # Dataset loading utilities
│   └── utils/                      # Shared utilities
└── .github/                        # Optional workflows and issue templates
```

All supported datasets use `ultralytics/cfg/models/WaferLO/WaferLO.yaml`; each dataset YAML supplies its own class names and split paths.
