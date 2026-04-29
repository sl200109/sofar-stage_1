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
ENV_STAGE5_OPEN6DOR_CKPT = "SOFAR_STAGE5_OPEN6DOR_CKPT"
ENV_STAGE5_SPATIALBENCH_CKPT = "SOFAR_STAGE5_SPATIALBENCH_CKPT"
ENV_STAGE5_OPEN6DOR_UPRIGHT_CKPT = "SOFAR_STAGE5_OPEN6DOR_UPRIGHT_CKPT"
ENV_STAGE5_OPEN6DOR_FLAT_CKPT = "SOFAR_STAGE5_OPEN6DOR_FLAT_CKPT"
ENV_STAGE5_OPEN6DOR_PLUG_CKPT = "SOFAR_STAGE5_OPEN6DOR_PLUG_CKPT"

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw_path: str | os.PathLike, base: Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if base is None:
        base = sofar_root()
    return (base / path).resolve()


def _optional_env_path(name: str, base: Path | None = None) -> Path | None:
    custom = os.getenv(name)
    if not custom:
        return None
    return _resolve_path(custom, base=base)


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


def stage5_open6dor_checkpoint_path() -> Path:
    custom = os.getenv(ENV_STAGE5_OPEN6DOR_CKPT)
    if custom:
        return _resolve_path(custom, base=sofar_root())
    return output_dir() / "stage5_open6dor_train" / "stage5_pilot_best.pth"


def stage5_spatialbench_checkpoint_path() -> Path:
    custom = os.getenv(ENV_STAGE5_SPATIALBENCH_CKPT)
    if custom:
        return _resolve_path(custom, base=sofar_root())
    return output_dir() / "stage5_spatialbench_train" / "stage5_pilot_best.pth"


def stage5_open6dor_upright_checkpoint_path() -> Path | None:
    return _optional_env_path(ENV_STAGE5_OPEN6DOR_UPRIGHT_CKPT, base=sofar_root())


def stage5_open6dor_flat_checkpoint_path() -> Path | None:
    return _optional_env_path(ENV_STAGE5_OPEN6DOR_FLAT_CKPT, base=sofar_root())


def stage5_open6dor_plug_checkpoint_path() -> Path | None:
    return _optional_env_path(ENV_STAGE5_OPEN6DOR_PLUG_CKPT, base=sofar_root())


def open6dor_dataset_dir() -> Path:
    return datasets_dir() / "open6dor_v2"


def open6dor_task_refine_6dof_dir() -> Path:
    return open6dor_dataset_dir() / "open6dor_v2" / "task_refine_6dof"


def spatialbench_dataset_dir() -> Path:
    return datasets_dir() / "6dof_spatialbench"
