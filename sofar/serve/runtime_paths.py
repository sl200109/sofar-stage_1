import os
from pathlib import Path


ENV_SOFAR_ROOT = "SOFAR_ROOT"
ENV_OUTPUT_DIR = "SOFAR_OUTPUT_DIR"
ENV_CHECKPOINTS_DIR = "SOFAR_CHECKPOINTS_DIR"
ENV_DATASETS_DIR = "SOFAR_DATASETS_DIR"
ENV_QWEN_PATH = "SOFAR_QWEN_PATH"
ENV_POINTSO_CFG = "SOFAR_POINTSO_CFG"
ENV_POINTSO_CKPT = "SOFAR_POINTSO_CKPT"
ENV_METRIC3D_CKPT = "SOFAR_METRIC3D_CKPT"
ENV_GROUNDINGDINO_TEXT_ENCODER = "SOFAR_GROUNDINGDINO_TEXT_ENCODER"

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw_path: str | os.PathLike, base: Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if base is None:
        base = sofar_root()
    return (base / path).resolve()


def sofar_root() -> Path:
    root = os.getenv(ENV_SOFAR_ROOT)
    if root:
        return _resolve_path(root, base=_DEFAULT_ROOT)
    return _DEFAULT_ROOT


def resolve_from_root(*parts: str) -> Path:
    return sofar_root().joinpath(*parts).resolve()


def checkpoints_dir() -> Path:
    custom = os.getenv(ENV_CHECKPOINTS_DIR)
    if custom:
        return _resolve_path(custom)
    return resolve_from_root("checkpoints")


def datasets_dir() -> Path:
    custom = os.getenv(ENV_DATASETS_DIR)
    if custom:
        return _resolve_path(custom)
    return resolve_from_root("datasets")


def output_dir() -> Path:
    custom = os.getenv(ENV_OUTPUT_DIR)
    if custom:
        return _resolve_path(custom)
    return resolve_from_root("output")


def ensure_output_dir() -> Path:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def qwen_checkpoint_path() -> Path:
    custom = os.getenv(ENV_QWEN_PATH)
    if custom:
        return _resolve_path(custom, base=checkpoints_dir())
    return checkpoints_dir() / "Qwen2.5-VL-3B"


def pointso_cfg_path() -> Path:
    custom = os.getenv(ENV_POINTSO_CFG)
    if custom:
        return _resolve_path(custom)
    return resolve_from_root("orientation", "cfgs", "train", "small.yaml")


def pointso_checkpoint_path() -> Path:
    custom = os.getenv(ENV_POINTSO_CKPT)
    if custom:
        return _resolve_path(custom, base=checkpoints_dir())
    return checkpoints_dir() / "small.pth"


def metric3d_checkpoint_path() -> Path:
    custom = os.getenv(ENV_METRIC3D_CKPT)
    if custom:
        return _resolve_path(custom, base=checkpoints_dir())
    return checkpoints_dir() / "metric_depth_vit_large_800k.pth"


def groundingdino_text_encoder_path() -> Path:
    custom = os.getenv(ENV_GROUNDINGDINO_TEXT_ENCODER)
    if custom:
        return _resolve_path(custom, base=checkpoints_dir())
    return checkpoints_dir() / "bert-base-uncased"


def open6dor_dataset_dir() -> Path:
    return datasets_dir() / "open6dor_v2"


def spatialbench_dataset_dir() -> Path:
    return datasets_dir() / "6dof_spatialbench"
