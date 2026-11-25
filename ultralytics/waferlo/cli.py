"""Task-specific entry points for the bundled WaferLO detector."""
import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import hashlib
from pathlib import Path

import yaml

# Repository root when running from the installed ``ultralytics`` package.
ROOT = Path(__file__).resolve().parents[2]
MODELS = {"waferlo": "WaferLO.yaml"}


def model_path(value):
    # Accept the short alias, its YAML spelling, and arbitrary custom paths.
    key = str(value).strip().lower()
    if key in MODELS or key in {f"{name}.yaml" for name in MODELS} or key in {f"{name}.yml" for name in MODELS}:
        import ultralytics
        return str(Path(ultralytics.__file__).parent / "cfg/models/WaferLO" / MODELS["waferlo"])
    return value


def resolve_data(value, split):
    """Normalize dataset paths without depending on global Ultralytics settings."""
    source = Path(value).expanduser().resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    base = Path(data.get("path", ".")).expanduser()
    if not base.is_absolute():
        base = (source.parent / base).resolve()
    data["path"] = str(base)
    for key in ("train", "val", "test"):
        if data.get(key):
            items = data[key] if isinstance(data[key], list) else [data[key]]
            paths = [str((base / item).resolve()) for item in items]
            data[key] = paths if isinstance(data[key], list) else paths[0]
    keys = ("train", "val") if split == "train" else (split,)
    for key in keys:
        if not data.get(key):
            raise ValueError(f"Dataset YAML requires {key!r}: {source}")
        for item in data[key] if isinstance(data[key], list) else [data[key]]:
            if not Path(item).exists():
                raise FileNotFoundError(f"Missing {key} split: {item}. Edit {source} first.")
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    target = Path("runs/data_configs") / (hashlib.sha256(text.encode()).hexdigest()[:16] + ".yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target.resolve())


def default_data():
    local = ROOT / "configs/datasets/wafer.yaml"
    if local.exists():
        return str(local)
    import ultralytics
    return str(Path(ultralytics.__file__).parent / "cfg/datasets/wafer.yaml")


@dataclass(frozen=True)
class TrainingRecipe:
    """Reference training recipe; values map directly to Ultralytics arguments."""

    optimizer: str = "auto"
    epochs: int = 300
    warmup_bias_lr: float = 0.0
    dfl: float = 1.5
    degrees: float = 0.0
    scale: float = 0.5
    mixup: float = 0.0
    copy_paste: float = 0.0
    close_mosaic: int = 10
    conf: float = 0.01
    patience: int = 300
    pretrain: int = 20
    save_period: int = 1

    @classmethod
    def from_namespace(cls, args):
        return cls(**{key: getattr(args, key) for key in cls.__dataclass_fields__})

    def validate(self):
        if self.epochs <= 0 or self.patience < 0 or self.close_mosaic < 0:
            raise ValueError("epochs must be positive; patience and close_mosaic must be nonnegative")
        for key in ("mixup", "copy_paste", "conf"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be within [0, 1]")
        for key in ("warmup_bias_lr", "dfl", "degrees", "scale"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.pretrain < -1:
            raise ValueError("pretrain must be -1 (validate every epoch) or nonnegative")
        if self.save_period != -1 and self.save_period <= 0:
            raise ValueError("save_period must be -1 (disabled) or positive")


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved, inspectable boundary between configuration and model execution."""

    command: str
    model: str
    arguments: Dict[str, Any]
    pretrained: Optional[str] = None

    def describe(self):
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


class ExperimentPlanner:
    """Resolve portable paths and assemble one explicit backend invocation."""

    @staticmethod
    def rooted(value, root):
        path = Path(value).expanduser()
        return str((root / path).resolve()) if not path.is_absolute() else str(path)

    def build(self, args):
        root = Path(args.path).expanduser().resolve()
        if args.imgsz <= 0:
            raise ValueError("imgsz must be positive")
        model = model_path(args.model)
        if args.model not in MODELS:
            model = self.rooted(model, root)
        options = dict(imgsz=args.imgsz, device=args.device)
        pretrained = None
        if args.command in ("train", "val", "predict"):
            options.update(project=self.rooted(args.project, root), name=args.name)
        if args.command in ("train", "val"):
            if args.batch == 0 or args.batch < -1 or args.workers < 0:
                raise ValueError("batch must be positive or -1; workers must be nonnegative")
            split = "train" if args.command == "train" else args.split
            options.update(data=resolve_data(self.rooted(args.data, root), split),
                           batch=args.batch, workers=args.workers)
        if args.command == "train":
            recipe = TrainingRecipe.from_namespace(args)
            recipe.validate()
            if args.resume and args.pretrained:
                raise ValueError("resume restores checkpoint state; omit --pretrained when resuming")
            if args.resume and Path(model).suffix != ".pt":
                raise ValueError("resume requires --model path/to/last.pt")
            if args.pretrained:
                pretrained = self.rooted(args.pretrained, root)
                if not Path(pretrained).is_file():
                    raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
            options.update(asdict(recipe), task="detect", seed=args.seed,
                           pretrained=False, resume=args.resume)
        elif args.command == "val":
            options.update(split=args.split, save_json=args.save_json)
        elif args.command == "predict":
            # Preserve URLs, cameras and stream sources understood by Ultralytics.
            options.update(source=args.source, conf=args.conf, save=True, stream=True)
        elif args.command == "export":
            options.update(format=args.format)
        else:
            options = dict(detailed=False, imgsz=args.imgsz)
        for item in args.override:
            key, value = parse_override(item)
            options[key] = value
        return ExecutionPlan(args.command, model, options, pretrained)


class ExperimentRunner:
    """Keep the original YAML -> optional weights -> model.train workflow."""

    def __init__(self, model_factory=None):
        self.model_factory = model_factory

    def run(self, plan):
        factory = self.model_factory
        if factory is None:
            from ultralytics import YOLO
            factory = YOLO
        model = factory(plan.model, task="detect")
        if plan.pretrained:
            model.load(plan.pretrained)
        if plan.command == "predict":
            for _ in model.predict(**plan.arguments):
                pass
            return None
        result = getattr(model, plan.command)(**plan.arguments)
        if plan.command == "val":
            print(result.results_dict)
        return result


def add_training_arguments(parser):
    """Expose the reference script's recipe without modifying framework defaults."""
    schedule = parser.add_argument_group("Training schedule and checkpoints")
    schedule.add_argument("--epochs", type=int, default=300)
    schedule.add_argument("--patience", type=int, default=300)
    schedule.add_argument("--save-period", "--save_period", dest="save_period", type=int, default=1)
    schedule.add_argument("--close-mosaic", "--close_mosaic", dest="close_mosaic", type=int, default=10)
    schedule.add_argument("--pretrain", type=int, default=20, help="Retain original validation delay; -1 validates every epoch")
    schedule.add_argument("--seed", type=int, default=0)
    schedule.add_argument("--resume", action="store_true", help="Restore state from --model path/to/last.pt")
    schedule.add_argument("--pretrained", "--pre_model", dest="pretrained", help="Explicit opt-in weight initialization")
    optimization = parser.add_argument_group("Optimization and detection loss")
    optimization.add_argument("--optimizer", default="auto")
    optimization.add_argument("--warmup-bias-lr", "--warmup_bias_lr", dest="warmup_bias_lr", type=float, default=0.0)
    optimization.add_argument("--dfl", type=float, default=1.5)
    optimization.add_argument("--conf", type=float, default=0.01)
    augmentation = parser.add_argument_group("Geometric and compositional augmentation")
    augmentation.add_argument("--degrees", type=float, default=0.0)
    augmentation.add_argument("--scale", type=float, default=0.5)
    augmentation.add_argument("--mixup", type=float, default=0.0)
    augmentation.add_argument("--copy-paste", "--copy_paste", dest="copy_paste", type=float, default=0.0)


def parse_override(value):
    """Parse an Ultralytics-style key=value override without limiting valid keys."""
    if "=" not in value:
        raise ValueError(f"override must use key=value: {value}")
    key, raw = value.split("=", 1)
    if not key.strip():
        raise ValueError("override key cannot be empty")
    try:
        return key.strip(), yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid override {value}: {exc}") from exc


def build_parser():
    parser = argparse.ArgumentParser(description="WaferLO experiment configuration and execution")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "val", "predict", "export", "info"):
        p = commands.add_parser(command, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        runtime = p.add_argument_group("Model and runtime")
        runtime.add_argument("--model", "--tarin_yaml", dest="model",
                             default="waferlo" if command in ("train", "info") else None,
                             required=command in ("val", "predict", "export"))
        runtime.add_argument("--path", default=str(ROOT), help="Base directory for relative model/data/output paths")
        runtime.add_argument("--imgsz", type=int, default=800)
        runtime.add_argument("--device", default="cpu", help="cpu, 0, or 0,1; never silently falls back to other GPUs")
        runtime.add_argument("--dry-run", action="store_true", help="Validate paths and print the resolved invocation without running a model")
        if command in ("train", "val"):
            data = p.add_argument_group("Dataset and loading")
            data.add_argument("--data", "--data_yaml", dest="data", default=default_data())
            data.add_argument("--batch", "--batch_size", dest="batch", type=int, default=24 if command == "train" else 16)
            data.add_argument("--workers", type=int, default=0)
        if command == "train":
            add_training_arguments(p)
        elif command == "val":
            p.add_argument("--split", choices=("val", "test"), default="test")
            p.add_argument("--save-json", action="store_true")
        elif command == "predict":
            p.add_argument("--source", required=True)
            p.add_argument("--conf", type=float, default=0.25)
        elif command == "export":
            p.add_argument("--format", default="onnx")
        if command in ("train", "val", "predict"):
            output = p.add_argument_group("Experiment output")
            output.add_argument("--project", "--save_dir", dest="project", default="runs/waferlo")
            output.add_argument("--name", "--modelName", dest="name", default=command)
        p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                       help="Pass any additional Ultralytics setting, repeatable (e.g. --override amp=True)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = ExperimentPlanner().build(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(plan.describe())
    if args.dry_run:
        return plan
    return ExperimentRunner().run(plan)


if __name__ == "__main__":
    main()
