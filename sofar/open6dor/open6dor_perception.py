import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve import pointso as orientation
from open6dor.utils import (
    build_orientation_template_hints,
    extract_open6dor_minimal_object_set,
    load_orientation_templates,
    normalize_object_name,
    preprocess_open6dor_image,
    resolve_orientation_template,
)
from serve.scene_graph import build_open6dor_lightweight_scene_graph, open6dor_scene_graph
from segmentation import sam, florence as detection
from serve.utils import generate_rotation_matrix, get_point_cloud_from_rgbd
from serve import runtime_paths
from serve.batch_logging import setup_timestamped_logging, write_json_outputs
from serve.agent_debug import summarize_agent_records, write_agent_trace_bundle
from serve.qwen_runtime import resolve_qwen_dtype
from serve.stage3_grounding import (
    constrain_child_mask_to_parent,
    crop_image_array,
    expand_roi_mask,
    first_detection_score,
    first_detection_xyxy,
    offset_xyxy,
    save_stage3_cache,
    serialize_xyxy,
)
from serve.stage4_point_data import (
    build_colored_points_from_mask,
    compute_geometry_priors,
    sample_points,
    save_stage4_cache,
)
from serve.stage5_inference import predict_from_stage4_dir
from serve.semantic_orientation_agent import (
    decide_auto_agent_route,
    decide_open6dor_agent_action,
    infer_open6dor_execution_band,
    verify_open6dor_agent_outcome,
)

warnings.filterwarnings("ignore")
open6dor_parsing_fn = None
open6dor_stage2_fast_parser_fn = None
open6dor_joint_reasoning_fn = None
open6dor_spatial_reasoning_fn = None
detection_model = None
sam_model = None
orientation_model = None
RUN_RECORDS = []
RUN_OPTIONS = {}
STAGE5_OPTIONS = {}
AGENT_OPTIONS = {}
ORIENTATION_TEMPLATES = {}
DEFAULT_PROGRESS_FILE = "open6dor_perception_progress.json"
DEFAULT_SUMMARY_FILE = "open6dor_perception_summary.json"
DEFAULT_STAGE5_OPEN6DOR_ALLOWED_MODES = {"upright", "upright_lens_forth"}
DEFAULT_STAGE5_EXPERT_ROUTING = "off"
STAGE5_TASK_FAMILY_UPRIGHT = "upright_vertical"
STAGE5_TASK_FAMILY_FLAT = "flat_upside_down_lying_flat"
STAGE5_TASK_FAMILY_PLUG = "plug_cap_sideways"
STAGE5_TASK_FAMILY_UNKNOWN = "unknown"
NAMED_PILOTS = {
    "open6dor10": ROOT_DIR / "open6dor" / "pilots" / "open6dor_pilot_10.json",
}


def resolve_llm_backend():
    backend = os.getenv("SOFAR_LLM_BACKEND", "qwen").strip().lower()
    if backend not in {"openai", "qwen"}:
        raise ValueError(f"Unsupported SOFAR_LLM_BACKEND: {backend}")
    return backend


def _env_flag(name, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_stage5_mode_label(value):
    label = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "lying_sideways": "lying_sideways",
        "lying_flat": "lying_flat",
        "clip_sideways": "clip_sideways",
        "upside_down_textual": "upside_down_textual",
        "sideways_textual": "sideways_textual",
        "upright_lens_forth": "upright_lens_forth",
    }
    return aliases.get(label, label)


def _parse_stage5_mode_list(raw_value, default_modes=None):
    if raw_value is None or str(raw_value).strip() == "":
        values = default_modes or set()
        return {str(item).strip().lower() for item in values if str(item).strip()}
    normalized = str(raw_value).strip().lower()
    if normalized in {"all", "*"}:
        return {"all"}
    return {
        _normalize_stage5_mode_label(item)
        for item in normalized.split(",")
        if _normalize_stage5_mode_label(item)
    }


def _optional_path_string(value):
    if value is None:
        return None
    raw = str(value).strip()
    return raw or None


def _safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_open6dor_stage5_task_family(orientation_mode):
    mode = _normalize_stage5_mode_label(orientation_mode)
    if mode in {"upright", "upright_lens_forth", "vertical"}:
        return STAGE5_TASK_FAMILY_UPRIGHT
    if mode in {"flat", "upside_down", "upside_down_textual", "lying_flat"}:
        return STAGE5_TASK_FAMILY_FLAT
    if (
        mode.startswith("plug_")
        or mode.startswith("cap_")
        or mode in {"sideways", "sideways_textual", "clip_sideways", "lying_sideways"}
    ):
        return STAGE5_TASK_FAMILY_PLUG
    return STAGE5_TASK_FAMILY_UNKNOWN


def resolve_open6dor_stage5_checkpoint_route(
    orientation_mode,
    *,
    parser_confidence=None,
    stage4_cache_available=False,
    object_score=None,
    part_score=None,
    fallback_required=False,
):
    routing_policy = str(STAGE5_OPTIONS.get("expert_routing") or DEFAULT_STAGE5_EXPERT_ROUTING).strip().lower()
    family = infer_open6dor_stage5_task_family(orientation_mode)
    shared_checkpoint = _optional_path_string(STAGE5_OPTIONS.get("checkpoint_path"))
    if not shared_checkpoint:
        shared_checkpoint = str(runtime_paths.stage5_open6dor_checkpoint_path())

    route = {
        "routing_policy": routing_policy,
        "task_family": family,
        "checkpoint_path": shared_checkpoint if routing_policy == "off" else None,
        "checkpoint_source": "shared_default" if routing_policy == "off" else "none",
        "route_reason": "expert_routing_off" if routing_policy == "off" else "",
        "signals": {
            "orientation_mode": _normalize_stage5_mode_label(orientation_mode),
            "task_family": family,
            "parser_confidence": _safe_float(parser_confidence),
            "stage4_cache_available": bool(stage4_cache_available),
            "object_score": _safe_float(object_score),
            "part_score": _safe_float(part_score),
            "fallback_required": bool(fallback_required),
        },
    }
    if routing_policy == "off":
        return route

    if fallback_required:
        route["route_reason"] = "fallback_required"
        return route
    if not stage4_cache_available:
        route["route_reason"] = "stage4_cache_missing"
        return route
    if family == STAGE5_TASK_FAMILY_UNKNOWN:
        route["route_reason"] = "unknown_task_family"
        return route

    expert_checkpoints = STAGE5_OPTIONS.get("expert_checkpoints") or {}
    selected_checkpoint = _optional_path_string(expert_checkpoints.get(family))
    if not selected_checkpoint:
        route["route_reason"] = f"missing_{family}_expert_checkpoint"
        return route

    route["checkpoint_path"] = selected_checkpoint
    route["checkpoint_source"] = "family_specific"
    if selected_checkpoint == shared_checkpoint:
        route["checkpoint_source"] = "family_default_shared"
    route["route_reason"] = f"{family}_expert"
    return route


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_dumps_safe(value):
    return json.dumps(_jsonify(value), ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pilot",
        type=str,
        default=None,
        help="Run a named fixed pilot set, e.g. --pilot open6dor10.",
    )
    group.add_argument(
        "--task-list",
        type=str,
        default=None,
        help="Run a JSON task list. Entries should be relative to the Open6DOR dataset root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N pending task directories after resume/skip filtering.",
    )
    parser.add_argument(
        "--speed-profile",
        choices=["off", "conservative"],
        default="off",
        help="Apply a named batch speed profile without changing result.json format.",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Ignore the current mode's progress file and start this run fresh.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Do not skip tasks that already have output/result.json during full-pipeline runs.",
    )
    parser.add_argument(
        "--stage2-parser-only",
        action="store_true",
        help="Run Stage 2 fast parser smoke only and export structured parser outputs.",
    )
    parser.add_argument(
        "--stage3-grounding-only",
        action="store_true",
        help="Run Stage 3 grounding skeleton only and export object/part caches.",
    )
    parser.add_argument(
        "--stage4-pointdata-only",
        action="store_true",
        help="Run Stage 4 point-data building from existing Stage 3 caches.",
    )
    parser.add_argument(
        "--use-stage5-head",
        action="store_true",
        help="Use the Stage 5 single-domain orientation head from an existing Stage 4 cache during full-pipeline inference.",
    )
    parser.add_argument(
        "--stage5-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path override for the Stage 5 orientation head.",
    )
    parser.add_argument(
        "--stage5-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for Stage 5 orientation head inference.",
    )
    parser.add_argument(
        "--stage5-num-points",
        type=int,
        default=1024,
        help="Number of points sampled before feeding the Stage 5 orientation head.",
    )
    parser.add_argument(
        "--stage5-open6dor-modes",
        type=str,
        default=None,
        help=(
            "Comma-separated Open6DOR orientation modes allowed to use Stage 5. "
            "Default is upright,upright_lens_forth. Use 'all' to disable mode gating."
        ),
    )
    parser.add_argument(
        "--stage5-expert-routing",
        choices=["off", "task_family"],
        default=DEFAULT_STAGE5_EXPERT_ROUTING,
        help=(
            "Optional Stage 5 checkpoint routing policy for Open6DOR. "
            "'task_family' routes by upright/flat/plug families and will not reuse "
            "the shared upright checkpoint for other families unless a family ckpt is configured."
        ),
    )
    parser.add_argument(
        "--stage5-upright-expert-checkpoint",
        type=str,
        default=None,
        help="Optional override for the upright/vertical expert checkpoint.",
    )
    parser.add_argument(
        "--stage5-flat-expert-checkpoint",
        type=str,
        default=None,
        help="Optional override for the flat/upside_down/lying_flat expert checkpoint.",
    )
    parser.add_argument(
        "--stage5-plug-expert-checkpoint",
        type=str,
        default=None,
        help="Optional override for the plug/cap/sideways expert checkpoint.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["off", "dataset", "auto"],
        default=None,
        help="Agent control mode. Defaults to dataset when --use-stage5-head is enabled, otherwise off.",
    )
    parser.add_argument(
        "--agent-policy",
        type=str,
        default="rule_v2",
        help="Agent policy label written into logs and debug traces.",
    )
    parser.add_argument(
        "--agent-save-trace",
        action="store_true",
        help="Write centralized agent traces to output/agent_debug/...",
    )
    parser.add_argument(
        "--agent-debug-dir",
        type=str,
        default=None,
        help="Optional debug root override. The final path will still be grouped as <root>/open6dor/<run_id>/.",
    )
    parser.add_argument(
        "--agent-shadow-eval",
        action="store_true",
        help="Run Stage 5 in shadow-only mode for baseline-only Open6DOR orientation bands.",
    )
    parser.add_argument(
        "--agent-extended-actions",
        action="store_true",
        help="Reserved hook for ROI rebuild / part-first fallback. Open6DOR v1 records the flag but does not execute extra actions.",
    )
    return parser.parse_args()


def load_qwen_model():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    qwen_path = str(runtime_paths.qwen_checkpoint_path())
    qwen_dtype = resolve_qwen_dtype(torch)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=qwen_dtype,
    )
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path,
        quantization_config=nf4_config,
        torch_dtype=qwen_dtype,
        attn_implementation="sdpa",
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(qwen_path)
    return qwen_model, processor


def sanitize_tag(value):
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def mode_files(artifact_tag=None):
    if artifact_tag:
        return (
            f"open6dor_perception_progress_{artifact_tag}.json",
            f"open6dor_perception_summary_{artifact_tag}.json",
            f"open6dor_perception_{artifact_tag}",
        )
    return DEFAULT_PROGRESS_FILE, DEFAULT_SUMMARY_FILE, "open6dor_perception"


def progress_file_path(output_dir, artifact_tag=None):
    progress_name, _, _ = mode_files(artifact_tag)
    return Path(output_dir) / progress_name


def latest_records(records):
    latest_by_task = {}
    for record in records:
        task_dir = record.get("task_dir")
        if not task_dir:
            continue
        latest_by_task[task_dir] = record
    return list(latest_by_task.values())


def summarize_numeric_field(records, field_name, statuses=("success",)):
    values = []
    for record in records:
        if record.get("status") not in statuses:
            continue
        value = record.get(field_name)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


def summarize_stage_timings(records):
    stage_fields = [
        "pcd_sec",
        "qwen_joint_sec",
        "parse_sec",
        "detection_sec",
        "sam_sec",
        "scene_graph_sec",
        "reasoning_sec",
        "total_sec",
    ]
    return {field: summarize_numeric_field(records, field) for field in stage_fields}


def save_progress(output_dir, run_id, dataset_root, dataset_paths, run_context):
    normalized_records = latest_records(RUN_RECORDS)
    success_records = [item for item in normalized_records if item.get("status") == "success"]
    summary = {
        "run_id": run_id,
        "dataset_root": str(dataset_root),
        "run_mode": run_context["run_mode"],
        "artifact_tag": run_context["artifact_tag"],
        "pilot": run_context.get("pilot"),
        "task_list": run_context.get("task_list"),
        "speed_profile": run_context["speed_profile"],
        "speed_profile_settings": run_context.get("speed_profile_settings", {}),
        "total_tasks": len(dataset_paths),
        "success_count": sum(1 for item in normalized_records if item["status"] == "success"),
        "error_count": sum(1 for item in normalized_records if item["status"] == "error"),
        "skipped_count": sum(1 for item in normalized_records if item["status"] == "skipped"),
        "processed_count": len(normalized_records),
        "remaining_count": max(0, len(dataset_paths) - len(normalized_records)),
        "avg_success_sec": summarize_numeric_field(normalized_records, "elapsed_sec")["mean"],
        "median_success_sec": summarize_numeric_field(normalized_records, "elapsed_sec")["median"],
        "min_success_sec": min((item.get("elapsed_sec") for item in success_records), default=None),
        "max_success_sec": max((item.get("elapsed_sec") for item in success_records), default=None),
        "stage_timing_summary": summarize_stage_timings(normalized_records),
        "agent_summary": summarize_open6dor_agent_records(normalized_records),
        "slowest_tasks": [
            {
                "task_dir": item.get("task_dir"),
                "status": item.get("status"),
                "total_sec": item.get("total_sec"),
                "failed_stage": item.get("failed_stage"),
                "error": item.get("error"),
            }
            for item in sorted(
                normalized_records,
                key=lambda row: row.get("total_sec") or row.get("elapsed_sec") or 0,
                reverse=True,
            )[:5]
        ],
        "records": normalized_records,
    }
    progress_path = progress_file_path(output_dir, run_context["artifact_tag"])
    tmp_path = progress_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2, ensure_ascii=False)
    tmp_path.replace(progress_path)
    return summary


def load_progress(output_dir, artifact_tag=None, reset_progress=False):
    if reset_progress:
        return [], set()
    progress_path = progress_file_path(output_dir, artifact_tag)
    if not progress_path.exists():
        return [], set()
    with progress_path.open("r", encoding="utf-8") as f:
        progress = json.load(f)
    records = latest_records(progress.get("records", []))
    completed = {item["task_dir"] for item in records if item.get("status") in {"success", "skipped"}}
    return records, completed


def should_skip_existing_result(task_dir, rerun_existing=False):
    if rerun_existing or os.getenv("SOFAR_OPEN6DOR_FORCE_RERUN", "").strip() == "1":
        return False
    result_path = os.path.join(task_dir, "output", "result.json")
    return os.path.exists(result_path)


def _stage_cache_force_rerun():
    return os.getenv("SOFAR_OPEN6DOR_FORCE_STAGE_RERUN", "").strip() == "1"


def _load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def has_completed_stage3_cache(task_dir):
    if _stage_cache_force_rerun():
        return False
    stage3_dir = Path(task_dir) / "output" / "stage3"
    cache_json = stage3_dir / "object_part_cache.json"
    object_mask_path = stage3_dir / "object_mask.npz"
    cache_payload = _load_json_if_exists(cache_json)
    if cache_payload is None or not object_mask_path.exists():
        return False
    grounding = cache_payload.get("grounding", {})
    return grounding.get("status") in {"success", "partial"}


def has_completed_stage4_cache(task_dir):
    if _stage_cache_force_rerun():
        return False
    stage4_dir = Path(task_dir) / "output" / "stage4"
    cache_json = stage4_dir / "point_data_cache.json"
    object_points_path = stage4_dir / "object_points.npz"
    cache_payload = _load_json_if_exists(cache_json)
    if cache_payload is None or not object_points_path.exists():
        return False
    return True


def filter_pending_stage_cache_tasks(dataset_paths, stage_name):
    if stage_name == "stage3":
        checker = has_completed_stage3_cache
    elif stage_name == "stage4":
        checker = has_completed_stage4_cache
    else:
        raise ValueError(f"Unsupported stage cache filter: {stage_name}")

    pending = []
    skipped = 0
    for task_dir in dataset_paths:
        if checker(task_dir):
            skipped += 1
            continue
        pending.append(task_dir)
    return pending, skipped


def _summarize_xyz_points(xyz_points):
    xyz_points = np.asarray(xyz_points, dtype=np.float32)
    if xyz_points.size == 0:
        raise ValueError("Cannot summarize empty xyz points")
    xyz_points = xyz_points.reshape(-1, 3)
    min_values = xyz_points.min(axis=0)
    max_values = xyz_points.max(axis=0)
    mean_values = xyz_points.mean(axis=0)
    center = [round(float(mean_values[0]), 2), round(float(mean_values[1]), 2), round(float(mean_values[2]), 2)]
    center_str = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
    bbox = [
        [float(min_values[0]), float(max_values[0])],
        [float(min_values[1]), float(max_values[1])],
        [float(min_values[2]), float(max_values[2])],
    ]
    bbox_str = {
        "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
        "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
        "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}",
    }
    return center, center_str, bbox, bbox_str


def _predict_cached_orientation(colored_object_pcd, direction_attributes, orientation_model):
    orientation_dict = {}
    attrs = [str(value).strip() for value in (direction_attributes or []) if str(value or "").strip()]
    if not attrs or colored_object_pcd is None or len(colored_object_pcd) == 0:
        return orientation_dict
    try:
        pred = orientation.pred_orientation(orientation_model, colored_object_pcd, attrs)
        for idx, attribute in enumerate(attrs):
            orientation_dict[attribute] = pred[idx]
    except Exception as exc:
        print(f"[open6dor] cached orientation fallback skipped: {exc}")
    return orientation_dict


def _build_other_objects_info_from_scene(image, pcd, mask, object_names):
    image_np = np.array(image)
    other_objects_info = []
    if mask is None:
        return other_objects_info
    for idx, object_name in enumerate(object_names):
        object_mask = mask[idx]
        segmented_object = pcd[object_mask]
        if segmented_object.size == 0:
            continue
        center, center_str, _, bbox_str = _summarize_xyz_points(segmented_object)
        node = {
            "object name": object_name,
            "center": center_str,
            "bounding box": bbox_str,
        }
        other_objects_info.append(node)
    return other_objects_info


def _build_cached_picked_object_fallback(task_dir, image, pcd, mask, object_names, parsed_info, orientation_model):
    stage4_dir = Path(task_dir) / "output" / "stage4"
    object_points_path = stage4_dir / "object_points.npz"
    if not object_points_path.exists():
        raise FileNotFoundError(f"Stage 4 object_points cache is missing at {object_points_path}")

    object_points = np.load(object_points_path)["points"].astype(np.float32)
    if object_points.size == 0:
        raise ValueError(f"Stage 4 object_points cache is empty at {object_points_path}")

    xyz_points = object_points[:, :3]
    center, center_str, bbox, bbox_str = _summarize_xyz_points(xyz_points)
    picked_object_name = str(parsed_info.get("picked_object", "")).strip() or "object"
    direction_attributes = parsed_info.get("direction_attributes", [])
    orientation_dict = _predict_cached_orientation(object_points, direction_attributes, orientation_model)

    picked_object_dict = {
        "object name": picked_object_name,
        "center": center,
        "bounding box": bbox,
        "orientation": orientation_dict,
    }
    picked_object_info = {
        "object name": picked_object_name,
        "center": center_str,
        "bounding box": bbox_str,
    }
    other_objects_info = _build_other_objects_info_from_scene(image, pcd, mask, object_names)
    return picked_object_info, other_objects_info, picked_object_dict


def discover_task_dirs(dataset_root):
    config_files = sorted(Path(dataset_root).rglob("task_config_new5.json"))
    task_dirs = []
    for config_file in config_files:
        path_str = str(config_file).replace("\\", "/")
        if "/task_refine_pos/" not in path_str and "/task_refine_rot/" not in path_str and "/task_refine_6dof/" not in path_str:
            continue
        task_dirs.append(str(config_file.parent))
    return sorted(set(task_dirs))


def resolve_task_list_path(args):
    if args.pilot:
        if args.pilot not in NAMED_PILOTS:
            raise ValueError(f"Unknown pilot set: {args.pilot}")
        return NAMED_PILOTS[args.pilot]
    if args.task_list:
        return Path(args.task_list)
    return None


def load_task_list(task_list_path, dataset_root):
    with Path(task_list_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Task list must be a JSON list: {task_list_path}")

    task_dirs = []
    missing = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"Task list entries must be strings: {item}")
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = Path(dataset_root) / item
        candidate = candidate.resolve()
        if candidate.exists():
            task_dirs.append(str(candidate))
        else:
            missing.append(item)
    if missing:
        raise FileNotFoundError(f"Missing task directories in task list: {missing}")
    return task_dirs


def build_run_context(args, dataset_root):
    task_list_path = resolve_task_list_path(args)
    if args.pilot:
        artifact_tag = sanitize_tag(args.pilot)
        run_mode = "pilot"
        rerun_existing = True
    elif args.task_list:
        artifact_tag = sanitize_tag(Path(args.task_list).stem)
        run_mode = "task_list"
        rerun_existing = True
    else:
        artifact_tag = None
        run_mode = "full"
        rerun_existing = bool(args.rerun_existing)

    progress_name, summary_name, log_prefix = mode_files(artifact_tag)
    dataset_paths = load_task_list(task_list_path, dataset_root) if task_list_path is not None else discover_task_dirs(dataset_root)
    return {
        "run_mode": run_mode,
        "artifact_tag": artifact_tag,
        "pilot": args.pilot,
        "task_list": str(task_list_path) if task_list_path else None,
        "progress_name": progress_name,
        "summary_name": summary_name,
        "log_prefix": log_prefix,
        "rerun_existing": rerun_existing,
        "dataset_paths": dataset_paths,
        "speed_profile": args.speed_profile,
    }


def apply_speed_profile(profile):
    settings = {}
    if profile == "conservative":
        defaults = {
            "SOFAR_SAVE_DEBUG_ARTIFACTS": "0",
            "SOFAR_SAM_PREFER_SINGLE_MASK": "1",
            "SOFAR_FLORENCE_MAX_NEW_TOKENS": "256",
            "SOFAR_FLORENCE_NUM_BEAMS": "1",
            "SOFAR_QWEN_OPEN6DOR_PARSE_MAX_NEW_TOKENS": "128",
            "SOFAR_QWEN_OPEN6DOR_REASON_MAX_NEW_TOKENS": "192",
            "SOFAR_QWEN_OPEN6DOR_JOINT_MAX_NEW_TOKENS": "256",
            "SOFAR_POINTSO_VOTE_NUM": "6",
            "SOFAR_POINTSO_SAMPLE_POINTS": "4096",
            "SOFAR_OPEN6DOR_USE_PERCEPTION_CACHE": "1",
        }
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
            settings[key] = os.environ[key]
    return settings


def perception_cache_path(task_dir):
    return Path(task_dir) / "output" / "perception_cache.json"


def mask_cache_path(task_dir):
    return Path(task_dir) / "output" / "mask_cache.npz"


def load_perception_cache(task_dir):
    cache_path = perception_cache_path(task_dir)
    mask_path = mask_cache_path(task_dir)
    if not cache_path.exists() or not mask_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as f:
        cache = json.load(f)
    with np.load(mask_path, allow_pickle=True) as npz:
        cache["mask"] = npz["mask"]
    return cache


def save_perception_cache(task_dir, cache_payload, mask):
    output_dir = Path(task_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = perception_cache_path(task_dir)
    mask_path = mask_cache_path(task_dir)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(cache_payload), f, indent=2, ensure_ascii=False)
    np.savez_compressed(mask_path, mask=mask)


def build_joint_object_context(task_info, prompt):
    target_name = normalize_object_name(task_info.get("target_obj_name", ""))
    context = extract_open6dor_minimal_object_set(prompt, target_name)
    template_hints = build_orientation_template_hints(context["picked_object"], ORIENTATION_TEMPLATES)
    context["direction_attributes"] = template_hints.get("direction_attributes", [])
    context["orientation_template_hints"] = template_hints
    if (
        "rot" in str(task_info.get("task_type", "")).lower()
        or "rotation" in prompt.lower()
        or "upright" in prompt.lower()
    ) and not template_hints.get("matched"):
        context["fallback_required"] = True
    return context


def build_perception_cache_payload(object_context, detections, mask, object_names, picked_object_info, other_objects_info, picked_object_dict):
    lightweight_scene_graph = build_open6dor_lightweight_scene_graph(
        picked_object_info,
        other_objects_info,
        picked_object_dict,
        target_orientation={attribute: picked_object_dict.get("orientation", {}).get(attribute) for attribute in object_context.get("direction_attributes", []) if attribute in picked_object_dict.get("orientation", {})},
    )
    return {
        "object_context": _jsonify(object_context),
        "object_names": list(object_names),
        "detections": {
            "xyxy": detections.xyxy.tolist(),
            "class_id": detections.class_id.tolist(),
            "confidence": detections.confidence.tolist(),
        },
        "picked_object_info": _jsonify(picked_object_info),
        "other_objects_info": _jsonify(other_objects_info),
        "picked_object_dict": _jsonify(picked_object_dict),
        "lightweight_scene_graph": _jsonify(lightweight_scene_graph),
        "mask_shape": list(mask.shape),
    }


def reuse_or_compute_perception(task_dir, preprocessed_image, pcd, object_context, output_path):
    use_cache = RUN_OPTIONS.get("use_perception_cache", True)
    cache = load_perception_cache(task_dir) if use_cache else None
    if cache is not None:
        return {
            "cache_hit": True,
            "mask": cache["mask"],
            "object_names": cache["object_names"],
            "picked_object_info": cache["picked_object_info"],
            "other_objects_info": cache["other_objects_info"],
            "picked_object_dict": cache["picked_object_dict"],
            "lightweight_scene_graph": cache["lightweight_scene_graph"],
            "detections": cache.get("detections", {}),
        }

    object_list = [object_context["picked_object"]] + list(object_context.get("related_objects", []))
    parsed_info = {
        "picked_object": object_context["picked_object"],
        "related_objects": object_context.get("related_objects", []),
        "direction_attributes": object_context.get("direction_attributes", []),
    }
    detections = detection.get_detections(
        preprocessed_image,
        object_list,
        detection_model,
        output_folder=output_path,
        single=True,
        save_artifacts=RUN_OPTIONS["save_debug_artifacts"],
    )
    mask, ann_img, object_names = sam.get_mask(
        preprocessed_image,
        object_list,
        sam_model,
        detections,
        output_folder=output_path,
        save_artifacts=RUN_OPTIONS["save_debug_artifacts"],
        prefer_single_mask=RUN_OPTIONS["prefer_single_mask"],
    )

    picked_object_info, other_objects_info, picked_object_dict = open6dor_scene_graph(
        preprocessed_image,
        pcd,
        mask,
        parsed_info,
        object_names,
        orientation_model,
        output_folder=output_path,
        save_debug_artifacts=RUN_OPTIONS["save_debug_artifacts"],
    )

    cache_payload = build_perception_cache_payload(
        object_context,
        detections,
        mask,
        object_names,
        picked_object_info,
        other_objects_info,
        picked_object_dict,
    )
    save_perception_cache(task_dir, cache_payload, mask)
    return {
        "cache_hit": False,
        "mask": mask,
        "object_names": object_names,
        "picked_object_info": picked_object_info,
        "other_objects_info": other_objects_info,
        "picked_object_dict": picked_object_dict,
        "lightweight_scene_graph": cache_payload["lightweight_scene_graph"],
        "detections": cache_payload["detections"],
    }


def infer_open6dor_stage5_mode(task_dir, task_info=None, parsed_info=None):
    task_info = task_info or {}
    parsed_info = parsed_info or {}
    candidates = [
        task_info.get("rot_tag_detail"),
        task_info.get("orientation_mode"),
        parsed_info.get("orientation_mode"),
    ]
    for candidate in candidates:
        label = _normalize_stage5_mode_label(candidate)
        if label:
            return label

    task_dir_str = str(task_dir).replace("\\", "/")
    match = re.search(r"\.__([^/]+?)(?:/|$)", task_dir_str)
    if match:
        return _normalize_stage5_mode_label(match.group(1))
    return ""


def should_apply_open6dor_stage5(task_dir, task_info=None, parsed_info=None):
    mode = infer_open6dor_stage5_mode(task_dir, task_info=task_info, parsed_info=parsed_info)
    allowed_modes = set(STAGE5_OPTIONS.get("allowed_modes") or [])
    if not STAGE5_OPTIONS.get("enabled"):
        return {"apply": False, "mode": mode, "reason": "stage5_disabled"}
    if "all" in allowed_modes:
        return {"apply": True, "mode": mode, "reason": "all_modes_enabled"}
    if not mode:
        return {"apply": False, "mode": "", "reason": "unknown_orientation_mode"}
    if mode in allowed_modes:
        return {"apply": True, "mode": mode, "reason": "mode_allowed"}
    return {"apply": False, "mode": mode, "reason": f"mode_not_allowed:{mode}"}


def resolve_agent_mode():
    return AGENT_OPTIONS.get("mode", "off")


def load_open6dor_stage4_signals(task_dir):
    stage4_dir = Path(task_dir) / "output" / "stage4"
    cache_path = stage4_dir / "point_data_cache.json"
    object_points_path = stage4_dir / "object_points.npz"
    payload = _load_json_if_exists(cache_path) or {}
    grounding = payload.get("grounding", {}) if isinstance(payload, dict) else {}
    geometry_priors = payload.get("geometry_priors", {}) if isinstance(payload, dict) else {}
    return {
        "stage4_dir": stage4_dir,
        "stage4_cache_available": cache_path.exists() and object_points_path.exists(),
        "object_score": grounding.get("object_score"),
        "part_score": grounding.get("part_score"),
        "part_ratio": geometry_priors.get("part_ratio"),
    }


def summarize_open6dor_agent_records(records):
    agent_records = [record for record in records if record.get("stage5_enabled")]
    summary = summarize_agent_records(agent_records)
    mode_distribution = {}
    family_distribution = {}
    source_distribution = {}
    semantic_status_distribution = {}
    upright_semantic_status_distribution = {}
    old_rule_pass_distribution = {}
    new_rule_pass_distribution = {}
    verifier_rule_version_distribution = {}
    verifier_decision_reason_distribution = {}
    target_axis_distribution = {}
    cosine_values = []
    for record in agent_records:
        mode = str(record.get("stage5_mode") or "none")
        mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
        family = str(record.get("stage5_checkpoint_family") or "none")
        family_distribution[family] = family_distribution.get(family, 0) + 1
        source = str(record.get("stage5_checkpoint_source") or "none")
        source_distribution[source] = source_distribution.get(source, 0) + 1
        diagnostics = record.get("stage5_orientation_diagnostics") or {}
        semantic_status = str(diagnostics.get("semantic_status") or "none")
        semantic_status_distribution[semantic_status] = semantic_status_distribution.get(semantic_status, 0) + 1
        if mode == "upright":
            upright_semantic_status_distribution[semantic_status] = upright_semantic_status_distribution.get(semantic_status, 0) + 1
        old_rule_pass = record.get("stage5_old_rule_pass")
        new_rule_pass = record.get("stage5_new_rule_pass")
        verifier_rule_version = str(record.get("stage5_verifier_rule_version") or "none")
        verifier_decision_reason = str(record.get("stage5_verifier_decision_reason") or "none")
        target_axis = record.get("stage5_target_axis")
        cosine_to_target_axis = record.get("stage5_cosine_to_target_axis")
        old_rule_pass_distribution[str(bool(old_rule_pass))] = old_rule_pass_distribution.get(str(bool(old_rule_pass)), 0) + 1
        new_rule_pass_distribution[str(bool(new_rule_pass))] = new_rule_pass_distribution.get(str(bool(new_rule_pass)), 0) + 1
        verifier_rule_version_distribution[verifier_rule_version] = verifier_rule_version_distribution.get(verifier_rule_version, 0) + 1
        verifier_decision_reason_distribution[verifier_decision_reason] = verifier_decision_reason_distribution.get(verifier_decision_reason, 0) + 1
        target_axis_key = json.dumps(target_axis, ensure_ascii=False, sort_keys=True) if target_axis is not None else "none"
        target_axis_distribution[target_axis_key] = target_axis_distribution.get(target_axis_key, 0) + 1
        if cosine_to_target_axis is not None:
            try:
                cosine_values.append(float(cosine_to_target_axis))
            except (TypeError, ValueError):
                pass
    summary["stage5_mode_distribution"] = mode_distribution
    summary["stage5_checkpoint_family_distribution"] = family_distribution
    summary["stage5_checkpoint_source_distribution"] = source_distribution
    summary["stage5_semantic_status_distribution"] = semantic_status_distribution
    summary["stage5_upright_semantic_status_distribution"] = upright_semantic_status_distribution
    summary["stage5_old_rule_pass_distribution"] = old_rule_pass_distribution
    summary["stage5_new_rule_pass_distribution"] = new_rule_pass_distribution
    summary["stage5_verifier_rule_version_distribution"] = verifier_rule_version_distribution
    summary["stage5_verifier_decision_reason_distribution"] = verifier_decision_reason_distribution
    summary["stage5_target_axis_distribution"] = target_axis_distribution
    if cosine_values:
        summary["stage5_cosine_to_target_axis_mean"] = round(sum(cosine_values) / len(cosine_values), 6)
    summary["stage5_applied_count"] = sum(1 for record in agent_records if record.get("stage5_applied"))
    return summary


def _attach_stage5_route_metadata(prediction, route):
    payload = dict(prediction or {})
    payload["checkpoint_route_policy"] = route.get("routing_policy")
    payload["checkpoint_family"] = route.get("task_family")
    payload["checkpoint_source"] = route.get("checkpoint_source")
    payload["checkpoint_route_reason"] = route.get("route_reason")
    payload["checkpoint_route_signals"] = route.get("signals", {})
    if route.get("checkpoint_path") and not payload.get("checkpoint_path"):
        payload["checkpoint_path"] = route.get("checkpoint_path")
    return payload


def _extract_stage5_orientation_diagnostics(stage5_prediction):
    diagnostics = dict((stage5_prediction or {}).get("orientation_diagnostics") or {})
    vector_components = diagnostics.get("vector_components") or {}
    diagnostics["vector_x"] = vector_components.get("x")
    diagnostics["vector_y"] = vector_components.get("y")
    diagnostics["vector_z"] = vector_components.get("z")
    return diagnostics


def _format_stage5_orientation_diagnostics(diagnostics):
    if not diagnostics:
        return ""
    return (
        f"status={diagnostics.get('semantic_status')} "
        f"attrs={diagnostics.get('requested_attributes', [])} "
        f"xyz=({diagnostics.get('vector_x')}, {diagnostics.get('vector_y')}, {diagnostics.get('vector_z')}) "
        f"top_up={diagnostics.get('top_up_score')} "
        f"bottom_down={diagnostics.get('bottom_down_score')} "
        f"warning={diagnostics.get('semantic_warning') or 'none'}"
    )


def process_dataset(task_dir):
    start_time = time.perf_counter()
    stage_times = {}
    failed_stage = None
    perception_cache_hit = False
    pipeline_mode = "legacy"
    stage5_prediction = None
    stage5_shadow_prediction = None
    stage5_checkpoint_route = {
        "routing_policy": str(STAGE5_OPTIONS.get("expert_routing") or DEFAULT_STAGE5_EXPERT_ROUTING),
        "task_family": STAGE5_TASK_FAMILY_UNKNOWN,
        "checkpoint_path": None,
        "checkpoint_source": "none",
        "route_reason": "stage5_disabled",
        "signals": {},
    }
    stage5_fallback_scene_graph_used = False
    stage5_gate = {"apply": False, "mode": "", "reason": "stage5_disabled"}
    agent_verification = {
        "verification_status": "not_run",
        "verification_reason": "verification_not_required",
        "final_decision": "skip_stage5_disabled",
        "final_decision_reason": "stage5_disabled",
        "selected_execution_mode": "baseline_only",
        "stage5_prediction_available": False,
        "target_orientation_available": False,
        "use_stage5": False,
        "use_orientation_prompt": False,
        "fallback_to_baseline": True,
        "shadow_used": False,
        "shadow_accepted": False,
        "triggered_reverification": False,
    }
    auto_route = None
    agent_decision = decide_open6dor_agent_action(
        stage5_enabled=False,
        orientation_mode="",
        stage5_gate_reason="stage5_disabled",
        fallback_required=False,
        parser_confidence=None,
        stage4_cache_available=False,
    )

    def record_stage(name, fn):
        nonlocal failed_stage
        failed_stage = name
        stage_start = time.perf_counter()
        try:
            result = fn()
            stage_times[f"{name}_sec"] = round(time.perf_counter() - stage_start, 2)
            return result
        except Exception:
            stage_times[f"{name}_sec"] = round(time.perf_counter() - stage_start, 2)
            raise

    def run_scene_graph_with_fallback(parsed_info_for_perception, image_for_scene, mask_for_scene, object_names_for_scene):
        nonlocal stage5_fallback_scene_graph_used
        try:
            return open6dor_scene_graph(
                image_for_scene,
                pcd,
                mask_for_scene,
                parsed_info_for_perception,
                object_names_for_scene,
                orientation_model,
                output_folder=output_path,
                save_debug_artifacts=RUN_OPTIONS["save_debug_artifacts"],
            )
        except ValueError as exc:
            if STAGE5_OPTIONS.get("enabled") and "is not in list" in str(exc):
                print(f"[open6dor] using Stage 4 cache fallback scene graph for {task_dir}: {exc}")
                stage5_fallback_scene_graph_used = True
                return _build_cached_picked_object_fallback(
                    task_dir,
                    image_for_scene,
                    pcd,
                    mask_for_scene,
                    object_names_for_scene,
                    parsed_info_for_perception,
                    orientation_model,
                )
            raise

    try:
        image_path = os.path.join(task_dir, "isaac_render-rgb-0-1.png")
        depth_path = os.path.join(task_dir, "isaac_render-depth-0-1.npy")
        with open(os.path.join(task_dir, "task_config_new5.json"), "r", encoding="utf-8") as f:
            task_info = json.load(f)
        prompt = (
            f"Pick the {task_info['target_obj_name']}. " + task_info["instruction"]
            if "rot" in task_dir
            else task_info["instruction"]
        )
        print(prompt)

        output_path = os.path.join(task_dir, "output")
        os.makedirs(output_path, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        depth = np.load(depth_path)
        vinvs = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-0.9028605, -0.0, 0.42993355, -0.0],
                [0.42993355, -0.0, 0.9028605, -0.0],
                [1.0, 0.0, 1.2, 1.0],
            ]
        )
        projs = [
            [1.7320507, 0.0, 0.0, 0.0],
            [0.0, 2.5980759, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.05, 0.0],
        ]
        pcd = record_stage(
            "pcd",
            lambda: get_point_cloud_from_rgbd(depth, np.array(image), vinvs, projs).cpu().numpy().astype(np.float64),
        )
        pcd = pcd.reshape(depth.shape[0], depth.shape[1], 6)[:, :, :3]
        preprocessed_image = preprocess_open6dor_image(image)

        def run_legacy_pipeline():
            parsed_info = record_stage("parse", lambda: open6dor_parsing_fn(prompt, image))
            parsed_info["related_objects"] = [] if "rot" in task_dir else parsed_info["related_objects"]
            object_list = [parsed_info["picked_object"]] + parsed_info["related_objects"]
            print(parsed_info)

            detections = record_stage(
                "detection",
                lambda: detection.get_detections(
                    preprocessed_image,
                    object_list,
                    detection_model,
                    output_folder=output_path,
                    single=True,
                    save_artifacts=RUN_OPTIONS["save_debug_artifacts"],
                ),
            )
            mask, _, object_names = record_stage(
                "sam",
                lambda: sam.get_mask(
                    preprocessed_image,
                    object_list,
                    sam_model,
                    detections,
                    output_folder=output_path,
                    save_artifacts=RUN_OPTIONS["save_debug_artifacts"],
                    prefer_single_mask=RUN_OPTIONS["prefer_single_mask"],
                ),
            )
            picked_object_info, other_objects_info, picked_object_dict = record_stage(
                "scene_graph",
                lambda: run_scene_graph_with_fallback(parsed_info, preprocessed_image, mask, object_names),
            )
            if "rot" not in task_dir:
                response = record_stage(
                    "reasoning",
                    lambda: open6dor_spatial_reasoning_fn(
                        preprocessed_image,
                        prompt,
                        picked_object_info,
                        other_objects_info,
                    ),
                )
                target_position = response["target_position"]
                print(response)
            else:
                stage_times["reasoning_sec"] = 0.0
                target_position = picked_object_dict["center"]
            return parsed_info, picked_object_dict, target_position

        parsed_info = None
        picked_object_dict = None
        target_position = None
        lightweight_scene_graph = None
        use_joint_path = llm_backend == "qwen"
        object_context = build_joint_object_context(task_info, prompt)
        if not object_context["picked_object"] or object_context["fallback_required"]:
            use_joint_path = False

        if use_joint_path:
            try:
                pipeline_mode = "joint"
                cache = load_perception_cache(task_dir) if RUN_OPTIONS["use_perception_cache"] else None
                if cache is not None:
                    perception_cache_hit = True
                    stage_times["detection_sec"] = 0.0
                    stage_times["sam_sec"] = 0.0
                    stage_times["scene_graph_sec"] = 0.0
                    picked_object_info = cache["picked_object_info"]
                    other_objects_info = cache["other_objects_info"]
                    picked_object_dict = cache["picked_object_dict"]
                    lightweight_scene_graph = cache["lightweight_scene_graph"]
                else:
                    parsed_info_for_perception = {
                        "picked_object": object_context["picked_object"],
                        "related_objects": object_context.get("related_objects", []),
                        "direction_attributes": object_context.get("direction_attributes", []),
                    }
                    object_list = [object_context["picked_object"]] + list(object_context.get("related_objects", []))
                    detections = record_stage(
                        "detection",
                        lambda: detection.get_detections(
                            preprocessed_image,
                            object_list,
                            detection_model,
                            output_folder=output_path,
                            single=True,
                            save_artifacts=RUN_OPTIONS["save_debug_artifacts"],
                        ),
                    )
                    mask, _, object_names = record_stage(
                        "sam",
                        lambda: sam.get_mask(
                            preprocessed_image,
                            object_list,
                            sam_model,
                            detections,
                            output_folder=output_path,
                            save_artifacts=RUN_OPTIONS["save_debug_artifacts"],
                            prefer_single_mask=RUN_OPTIONS["prefer_single_mask"],
                        ),
                    )
                    picked_object_info, other_objects_info, picked_object_dict = record_stage(
                        "scene_graph",
                        lambda: run_scene_graph_with_fallback(parsed_info_for_perception, preprocessed_image, mask, object_names),
                    )
                    lightweight_scene_graph = build_open6dor_lightweight_scene_graph(
                        picked_object_info,
                        other_objects_info,
                        picked_object_dict,
                        target_orientation={
                            attribute: picked_object_dict.get("orientation", {}).get(attribute)
                            for attribute in object_context.get("direction_attributes", [])
                            if attribute in picked_object_dict.get("orientation", {})
                        },
                    )
                    if RUN_OPTIONS["use_perception_cache"]:
                        save_perception_cache(
                            task_dir,
                            build_perception_cache_payload(
                                object_context,
                                detections,
                                mask,
                                object_names,
                                picked_object_info,
                                other_objects_info,
                                picked_object_dict,
                            ),
                            mask,
                        )

                joint_info = record_stage(
                    "qwen_joint",
                    lambda: open6dor_joint_reasoning_fn(
                        preprocessed_image,
                        prompt,
                        lightweight_scene_graph,
                        {
                            "picked_object": object_context["picked_object"],
                            "related_objects": object_context.get("related_objects", []),
                            "relation_type": object_context.get("relation_type", "none"),
                        },
                        object_context.get("orientation_template_hints", {}),
                    ),
                )
                parsed_info = {
                    "picked_object": joint_info.get("picked_object") or object_context["picked_object"],
                    "related_objects": joint_info.get("related_objects") or object_context.get("related_objects", []),
                    "direction_attributes": joint_info.get("direction_attributes") or object_context.get("direction_attributes", []),
                    "target_orientation": joint_info.get("target_orientation", {}),
                }
                target_position = picked_object_dict["center"] if "rot" in task_dir else joint_info["target_position"]
                stage_times["parse_sec"] = 0.0
                stage_times["reasoning_sec"] = 0.0
            except Exception as joint_exc:
                print(f"[open6dor] joint pipeline failed, falling back to legacy path: {joint_exc}")
                pipeline_mode = "legacy_fallback"
                stage_times.pop("qwen_joint_sec", None)
                parsed_info, picked_object_dict, target_position = run_legacy_pipeline()
        else:
            pipeline_mode = "legacy"
            parsed_info, picked_object_dict, target_position = run_legacy_pipeline()

        parsed_info.setdefault("orientation_mode", infer_open6dor_stage5_mode(task_dir, task_info=task_info, parsed_info=parsed_info))
        init_position = picked_object_dict["center"]
        init_orientation = picked_object_dict["orientation"]
        target_orientation = parsed_info["target_orientation"]

        if STAGE5_OPTIONS.get("enabled"):
            stage5_start = time.perf_counter()
            stage4_signals = load_open6dor_stage4_signals(task_dir)
            agent_mode = resolve_agent_mode()
            stage5_mode = infer_open6dor_stage5_mode(task_dir, task_info=task_info, parsed_info=parsed_info)

            try:
                if agent_mode == "off":
                    stage5_gate = should_apply_open6dor_stage5(task_dir, task_info=task_info, parsed_info=parsed_info)
                    if not stage5_gate["apply"]:
                        stage5_checkpoint_route = resolve_open6dor_stage5_checkpoint_route(
                            stage5_gate.get("mode", ""),
                            parser_confidence=(parsed_info or {}).get("parser_confidence"),
                            stage4_cache_available=bool(stage4_signals.get("stage4_cache_available")),
                            object_score=stage4_signals.get("object_score"),
                            part_score=stage4_signals.get("part_score"),
                            fallback_required=bool(object_context.get("fallback_required")),
                        )
                        stage5_prediction = _attach_stage5_route_metadata(
                            {
                            "status": "skipped",
                            "orientation_mode": stage5_gate["mode"],
                            "skip_reason": stage5_gate["reason"],
                            },
                            stage5_checkpoint_route,
                        )
                        agent_decision = {
                            "dataset": "open6dor",
                            "controller": "disabled",
                            "policy_version": AGENT_OPTIONS.get("policy", "rule_v2"),
                            "task_type": "manipulation",
                            "question_type_or_orientation_mode": stage5_gate.get("mode", ""),
                            "applicable": False,
                            "decision": "skip_stage5_due_to_mode_gating",
                            "decision_reason": stage5_gate.get("reason"),
                            "verification_status": "not_run",
                            "selected_execution_mode": "baseline_only",
                            "selected_actions": ["fallback_to_baseline_reasoning"],
                            "stage5_allowed": False,
                            "use_orientation_prompt": False,
                            "fallback_to_baseline": True,
                            "agent_signals": {
                                **stage4_signals,
                                "orientation_mode": stage5_gate.get("mode", ""),
                                "legacy_mode_gating": True,
                                "stage5_task_family": stage5_checkpoint_route.get("task_family"),
                                "stage5_checkpoint_route_reason": stage5_checkpoint_route.get("route_reason"),
                                "stage5_checkpoint_source": stage5_checkpoint_route.get("checkpoint_source"),
                            },
                        }
                        print(
                            f"[open6dor] stage5 skipped mode={stage5_gate['mode'] or 'unknown'} "
                            f"reason={stage5_gate['reason']}"
                        )
                    else:
                        stage5_checkpoint_route = resolve_open6dor_stage5_checkpoint_route(
                            stage5_mode,
                            parser_confidence=(parsed_info or {}).get("parser_confidence"),
                            stage4_cache_available=bool(stage4_signals.get("stage4_cache_available")),
                            object_score=stage4_signals.get("object_score"),
                            part_score=stage4_signals.get("part_score"),
                            fallback_required=bool(object_context.get("fallback_required")),
                        )
                        selected_checkpoint = stage5_checkpoint_route.get("checkpoint_path")
                        if selected_checkpoint:
                            print(
                                f"[open6dor] stage5 checkpoint route family={stage5_checkpoint_route.get('task_family')} "
                                f"source={stage5_checkpoint_route.get('checkpoint_source')} "
                                f"reason={stage5_checkpoint_route.get('route_reason')}"
                            )
                            stage5_prediction = _attach_stage5_route_metadata(
                                predict_from_stage4_dir(
                                    stage4_signals["stage4_dir"],
                                    dataset="open6dor",
                                    checkpoint_path=selected_checkpoint,
                                    device=STAGE5_OPTIONS.get("device", "auto"),
                                    num_points=STAGE5_OPTIONS.get("num_points", 1024),
                                ),
                                stage5_checkpoint_route,
                            )
                            diag_line = _format_stage5_orientation_diagnostics(
                                _extract_stage5_orientation_diagnostics(stage5_prediction)
                            )
                            if diag_line:
                                print(f"[open6dor] stage5 diagnostics {diag_line}")
                        else:
                            print(
                                f"[open6dor] stage5 route skipped family={stage5_checkpoint_route.get('task_family')} "
                                f"reason={stage5_checkpoint_route.get('route_reason')}"
                            )
                            stage5_prediction = _attach_stage5_route_metadata(
                                {
                                    "status": "skipped",
                                    "orientation_mode": stage5_mode,
                                    "skip_reason": stage5_checkpoint_route.get("route_reason"),
                                },
                                stage5_checkpoint_route,
                            )
                        if stage5_prediction and stage5_prediction.get("target_orientation"):
                            target_orientation = stage5_prediction["target_orientation"]
                            parsed_info["target_orientation"] = target_orientation
                            print(f"[open6dor] stage5 target_orientation={target_orientation}")
                            if lightweight_scene_graph is not None:
                                lightweight_scene_graph["stage5_head"] = stage5_prediction
                                if isinstance(lightweight_scene_graph.get("picked_object"), dict):
                                    lightweight_scene_graph["picked_object"]["stage5_head"] = stage5_prediction
                                    lightweight_scene_graph["picked_object"]["target_orientation"] = target_orientation
                        agent_decision = {
                            "dataset": "open6dor",
                            "controller": "disabled",
                            "policy_version": AGENT_OPTIONS.get("policy", "rule_v2"),
                            "task_type": "manipulation",
                            "question_type_or_orientation_mode": stage5_mode,
                            "applicable": bool(stage5_prediction and stage5_prediction.get("target_orientation")),
                            "decision": (
                                "direct_stage5_injection"
                                if stage5_prediction and stage5_prediction.get("target_orientation")
                                else "skip_stage5_due_to_checkpoint_routing"
                            ),
                            "decision_reason": (
                                "agent_mode_off"
                                if stage5_prediction and stage5_prediction.get("target_orientation")
                                else stage5_checkpoint_route.get("route_reason", "agent_mode_off")
                            ),
                            "verification_status": "accepted" if stage5_prediction and stage5_prediction.get("target_orientation") else "rejected",
                            "selected_execution_mode": "stage5_direct_verified" if stage5_prediction and stage5_prediction.get("target_orientation") else "baseline_only",
                            "selected_actions": (
                                ["run_stage5", "inject_scene_graph"]
                                if stage5_prediction and stage5_prediction.get("target_orientation")
                                else ["fallback_to_baseline_reasoning"]
                            ),
                            "stage5_allowed": bool(stage5_prediction and stage5_prediction.get("target_orientation")),
                            "use_orientation_prompt": False,
                            "fallback_to_baseline": not bool(stage5_prediction and stage5_prediction.get("target_orientation")),
                            "agent_signals": {
                                **stage4_signals,
                                "orientation_mode": stage5_mode,
                                "legacy_mode_gating": True,
                                "stage5_task_family": stage5_checkpoint_route.get("task_family"),
                                "stage5_checkpoint_route_reason": stage5_checkpoint_route.get("route_reason"),
                                "stage5_checkpoint_source": stage5_checkpoint_route.get("checkpoint_source"),
                            },
                        }
                    agent_verification = {
                        "verification_status": "accepted" if stage5_prediction and stage5_prediction.get("target_orientation") else "not_run",
                        "verification_reason": (
                            "legacy_direct_stage5"
                            if stage5_prediction and stage5_prediction.get("target_orientation")
                            else stage5_checkpoint_route.get("route_reason", "verification_not_required")
                        ),
                        "final_decision": agent_decision.get("decision"),
                        "final_decision_reason": agent_decision.get("decision_reason"),
                        "selected_execution_mode": agent_decision.get("selected_execution_mode", "baseline_only"),
                        "stage5_prediction_available": bool(stage5_prediction),
                        "target_orientation_available": bool((stage5_prediction or {}).get("target_orientation")),
                        "use_stage5": bool(stage5_prediction and stage5_prediction.get("target_orientation")),
                        "use_orientation_prompt": False,
                        "fallback_to_baseline": not bool(stage5_prediction and stage5_prediction.get("target_orientation")),
                        "shadow_used": False,
                        "shadow_accepted": False,
                        "triggered_reverification": False,
                    }
                    auto_route = {
                        "controller": "agent_mode_off",
                        "policy_version": AGENT_OPTIONS.get("policy", "rule_v2"),
                        "dataset_hint": "open6dor",
                        "selected_dataset_agent": "disabled",
                        "selected_execution_mode": agent_verification.get("selected_execution_mode", "baseline_only"),
                        "route_reason": "agent_mode_off",
                        "child_decision": agent_decision.get("decision"),
                        "child_verification_status": agent_verification.get("verification_status", "not_run"),
                    }
                else:
                    execution_band = infer_open6dor_execution_band(stage5_mode)
                    stage5_gate = {
                        "apply": execution_band in {"direct_allow", "conditional_verify"},
                        "mode": stage5_mode,
                        "reason": f"{execution_band}_mode",
                    }
                    agent_decision = decide_open6dor_agent_action(
                        stage5_enabled=bool(STAGE5_OPTIONS.get("enabled")),
                        orientation_mode=stage5_mode,
                        stage5_gate_reason=stage5_gate.get("reason", ""),
                        fallback_required=bool(object_context.get("fallback_required")),
                        parser_confidence=(parsed_info or {}).get("parser_confidence"),
                        stage4_cache_available=bool(stage4_signals.get("stage4_cache_available")),
                        object_score=stage4_signals.get("object_score"),
                        part_score=stage4_signals.get("part_score"),
                        part_ratio=stage4_signals.get("part_ratio"),
                        stage5_fallback_scene_graph_used=stage5_fallback_scene_graph_used,
                        shadow_enabled=bool(AGENT_OPTIONS.get("shadow_eval")),
                    )
                    print(
                        f"[open6dor] agent decision={agent_decision.get('decision')} "
                        f"reason={agent_decision.get('decision_reason')}"
                    )

                    stage5_checkpoint_route = resolve_open6dor_stage5_checkpoint_route(
                        stage5_mode,
                        parser_confidence=(parsed_info or {}).get("parser_confidence"),
                        stage4_cache_available=bool(stage4_signals.get("stage4_cache_available")),
                        object_score=stage4_signals.get("object_score"),
                        part_score=stage4_signals.get("part_score"),
                        fallback_required=bool(object_context.get("fallback_required")),
                    )
                    agent_decision["agent_signals"] = {
                        **(agent_decision.get("agent_signals") or {}),
                        "stage5_task_family": stage5_checkpoint_route.get("task_family"),
                        "stage5_checkpoint_route_reason": stage5_checkpoint_route.get("route_reason"),
                        "stage5_checkpoint_source": stage5_checkpoint_route.get("checkpoint_source"),
                    }

                    should_run_stage5 = agent_decision.get("decision") in {
                        "use_stage5_direct",
                        "use_stage5_conditional_verify",
                        "shadow_stage5_for_debug",
                    } and bool(stage5_checkpoint_route.get("checkpoint_path"))
                    if should_run_stage5:
                        print(
                            f"[open6dor] stage5 checkpoint route family={stage5_checkpoint_route.get('task_family')} "
                            f"source={stage5_checkpoint_route.get('checkpoint_source')} "
                            f"reason={stage5_checkpoint_route.get('route_reason')}"
                        )
                        stage5_prediction = _attach_stage5_route_metadata(
                            predict_from_stage4_dir(
                                stage4_signals["stage4_dir"],
                                dataset="open6dor",
                                checkpoint_path=stage5_checkpoint_route.get("checkpoint_path"),
                                device=STAGE5_OPTIONS.get("device", "auto"),
                                num_points=STAGE5_OPTIONS.get("num_points", 1024),
                            ),
                            stage5_checkpoint_route,
                        )
                        diag_line = _format_stage5_orientation_diagnostics(
                            _extract_stage5_orientation_diagnostics(stage5_prediction)
                        )
                        if diag_line:
                            print(f"[open6dor] stage5 diagnostics {diag_line}")
                    else:
                        route_skip_reason = stage5_checkpoint_route.get("route_reason")
                        if agent_decision.get("decision") in {
                            "use_stage5_direct",
                            "use_stage5_conditional_verify",
                            "shadow_stage5_for_debug",
                        }:
                            print(
                                f"[open6dor] stage5 route skipped family={stage5_checkpoint_route.get('task_family')} "
                                f"reason={route_skip_reason}"
                            )
                        stage5_prediction = _attach_stage5_route_metadata(
                            {
                                "status": "skipped",
                                "orientation_mode": stage5_mode,
                                "skip_reason": route_skip_reason or agent_decision.get("decision_reason"),
                            },
                            stage5_checkpoint_route,
                        )

                    agent_verification = verify_open6dor_agent_outcome(
                        agent_decision,
                        prediction=stage5_prediction if should_run_stage5 else None,
                        orientation_mode=stage5_mode,
                        stage4_cache_available=bool(stage4_signals.get("stage4_cache_available")),
                        target_orientation=(stage5_prediction or {}).get("target_orientation"),
                        direction_attributes=(stage5_prediction or {}).get("direction_attributes"),
                    )
                    print(
                        f"[open6dor] agent verify={agent_verification.get('verification_status')} "
                        f"reason={agent_verification.get('verification_reason')}"
                    )

                    if agent_verification.get("shadow_used"):
                        stage5_shadow_prediction = stage5_prediction
                        print(
                            f"[open6dor] stage5 shadow mode={stage5_mode or 'unknown'} "
                            f"accepted={agent_verification.get('shadow_accepted')}"
                        )
                    elif agent_verification.get("use_stage5") and stage5_prediction and stage5_prediction.get("target_orientation"):
                        target_orientation = stage5_prediction["target_orientation"]
                        parsed_info["target_orientation"] = target_orientation
                        print(f"[open6dor] stage5 target_orientation={target_orientation}")
                        if lightweight_scene_graph is not None:
                            lightweight_scene_graph["stage5_head"] = stage5_prediction
                            if isinstance(lightweight_scene_graph.get("picked_object"), dict):
                                lightweight_scene_graph["picked_object"]["stage5_head"] = stage5_prediction
                                lightweight_scene_graph["picked_object"]["target_orientation"] = target_orientation
                    else:
                        if not stage5_prediction or stage5_prediction.get("status") == "skipped":
                            print(
                                f"[open6dor] stage5 skipped mode={stage5_mode or 'unknown'} "
                                f"reason={agent_decision.get('decision_reason')}"
                            )
                        else:
                            print(
                                f"[open6dor] stage5 rejected mode={stage5_mode or 'unknown'} "
                                f"reason={agent_verification.get('verification_reason')}"
                            )

                    if agent_mode == "auto":
                        auto_route = decide_auto_agent_route(
                            task_dir=task_dir,
                            orientation_mode=stage5_mode,
                            child_decision={
                                **agent_decision,
                                "verification_status": agent_verification.get("verification_status"),
                                "selected_execution_mode": agent_verification.get(
                                    "selected_execution_mode",
                                    agent_decision.get("selected_execution_mode"),
                                ),
                            },
                        )
                    else:
                        auto_route = {
                            "controller": "dataset_mode_fixed",
                            "policy_version": AGENT_OPTIONS.get("policy", "rule_v2"),
                            "dataset_hint": "open6dor",
                            "selected_dataset_agent": agent_decision.get("controller"),
                            "selected_execution_mode": agent_verification.get(
                                "selected_execution_mode",
                                agent_decision.get("selected_execution_mode", "baseline_only"),
                            ),
                            "route_reason": "dataset_mode_fixed",
                            "child_decision": agent_verification.get(
                                "final_decision",
                                agent_decision.get("decision"),
                            ),
                            "child_verification_status": agent_verification.get("verification_status", "not_run"),
                        }
                stage_times["stage5_head_sec"] = round(time.perf_counter() - stage5_start, 2)
            except Exception as exc:
                stage5_prediction = _attach_stage5_route_metadata(
                    {
                        "status": "error",
                        "error": str(exc),
                        "orientation_mode": stage5_mode,
                        "skip_reason": str(exc),
                    },
                    stage5_checkpoint_route,
                )
                stage_times["stage5_head_sec"] = round(time.perf_counter() - stage5_start, 2)
                print(f"[open6dor] stage5 head skipped for {task_dir}: {exc}")

        common_directions = [direction for direction in target_orientation.keys() if direction in init_orientation]
        if common_directions:
            init_directions = [init_orientation[direction] for direction in common_directions]
            target_directions = [target_orientation[direction] for direction in common_directions]
            transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
        else:
            transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        result = {
            "task_dir": task_dir,
            "init_position": init_position,
            "target_position": target_position,
            "delta_position": [round(target_position[i] - init_position[i], 2) for i in range(3)],
            "init_orientation": init_orientation,
            "target_orientation": target_orientation,
            "transform_matrix": transform_matrix,
            "stage5_head": stage5_prediction,
        }
        with open(os.path.join(output_path, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print("Successfully saved result for", output_path)
        elapsed = round(time.perf_counter() - start_time, 2)
        stage5_orientation_diagnostics = _extract_stage5_orientation_diagnostics(stage5_prediction)
        return {
            "task_dir": task_dir,
            "sample_key": task_dir,
            "status": "success",
            "result_file": os.path.join(output_path, "result.json"),
            "elapsed_sec": elapsed,
            "total_sec": elapsed,
            "perception_cache_hit": perception_cache_hit,
            "pipeline_mode": pipeline_mode,
            "stage5_enabled": bool(STAGE5_OPTIONS.get("enabled")),
            "agent_controller": agent_decision.get("controller"),
            "agent_policy": agent_decision.get("policy_version", AGENT_OPTIONS.get("policy", "rule_v2")),
            "agent_selected_dataset_agent": (auto_route or {}).get("selected_dataset_agent", "disabled"),
            "agent_route_reason": (auto_route or {}).get("route_reason", "agent_mode_off"),
            "agent_decision": agent_verification.get("final_decision", agent_decision.get("decision")),
            "agent_decision_reason": agent_verification.get("final_decision_reason", agent_decision.get("decision_reason")),
            "agent_verification_status": agent_verification.get("verification_status"),
            "agent_verification_reason": agent_verification.get("verification_reason"),
            "agent_selected_execution_mode": agent_verification.get(
                "selected_execution_mode",
                agent_decision.get("selected_execution_mode"),
            ),
            "agent_stage5_allowed": agent_decision.get("stage5_allowed"),
            "agent_used_stage5": bool(agent_verification.get("use_stage5")),
            "agent_fallback_to_baseline": agent_verification.get("fallback_to_baseline", True),
            "agent_triggered_reverification": bool(agent_verification.get("triggered_reverification")),
            "agent_shadow_used": bool(agent_verification.get("shadow_used")),
            "agent_shadow_accepted": bool(agent_verification.get("shadow_accepted")),
            "agent_signals": agent_decision.get("agent_signals", {}),
            "stage5_applied": bool(agent_verification.get("use_stage5")),
            "stage5_checkpoint": (stage5_prediction or {}).get("checkpoint_path"),
            "stage5_checkpoint_family": (stage5_prediction or {}).get("checkpoint_family", stage5_checkpoint_route.get("task_family")),
            "stage5_checkpoint_source": (stage5_prediction or {}).get("checkpoint_source", stage5_checkpoint_route.get("checkpoint_source")),
            "stage5_checkpoint_route_reason": (stage5_prediction or {}).get("checkpoint_route_reason", stage5_checkpoint_route.get("route_reason")),
            "stage5_checkpoint_route_policy": (stage5_prediction or {}).get("checkpoint_route_policy", stage5_checkpoint_route.get("routing_policy")),
            "stage5_checkpoint_route_signals": (stage5_prediction or {}).get("checkpoint_route_signals", stage5_checkpoint_route.get("signals", {})),
            "stage5_mode": stage5_gate.get("mode"),
            "stage5_skip_reason": (stage5_prediction or {}).get("skip_reason", ""),
            "stage5_raw_direction_attributes": (stage5_prediction or {}).get("raw_direction_attributes", []),
            "stage5_direction_attributes": (stage5_prediction or {}).get("direction_attributes", []),
            "stage5_direction_vector": (stage5_prediction or {}).get("direction_vector"),
            "stage5_target_orientation": (stage5_prediction or {}).get("target_orientation"),
            "stage5_orientation_diagnostics": stage5_orientation_diagnostics,
            "stage5_vector_x": stage5_orientation_diagnostics.get("vector_x"),
            "stage5_vector_y": stage5_orientation_diagnostics.get("vector_y"),
            "stage5_vector_z": stage5_orientation_diagnostics.get("vector_z"),
            "stage5_top_up_score": stage5_orientation_diagnostics.get("top_up_score"),
            "stage5_bottom_down_score": stage5_orientation_diagnostics.get("bottom_down_score"),
            "stage5_semantic_status": stage5_orientation_diagnostics.get("semantic_status"),
            "stage5_semantic_warning": stage5_orientation_diagnostics.get("semantic_warning"),
            "stage5_old_rule_pass": agent_verification.get("old_rule_pass"),
            "stage5_new_rule_pass": agent_verification.get("new_rule_pass"),
            "stage5_target_axis": agent_verification.get("target_axis"),
            "stage5_target_axis_label": agent_verification.get("target_axis_label"),
            "stage5_target_axis_source": agent_verification.get("target_axis_source"),
            "stage5_cosine_to_target_axis": agent_verification.get("cosine_to_target_axis"),
            "stage5_verifier_rule_version": agent_verification.get("verifier_rule_version"),
            "stage5_verifier_decision_reason": agent_verification.get("verifier_decision_reason"),
            "stage5_head": stage5_prediction,
            "stage5_shadow_head": stage5_shadow_prediction,
            "stage5_fallback_scene_graph_used": stage5_fallback_scene_graph_used,
            **stage_times,
        }

    except Exception as exc:
        elapsed = round(time.perf_counter() - start_time, 2)
        print(f"Error processing {task_dir}: {exc}")
        stage5_orientation_diagnostics = _extract_stage5_orientation_diagnostics(stage5_prediction)
        return {
            "task_dir": task_dir,
            "sample_key": task_dir,
            "status": "error",
            "error": str(exc),
            "elapsed_sec": elapsed,
            "total_sec": elapsed,
            "failed_stage": failed_stage,
            "perception_cache_hit": perception_cache_hit,
            "pipeline_mode": pipeline_mode,
            "stage5_enabled": bool(STAGE5_OPTIONS.get("enabled")),
            "agent_controller": agent_decision.get("controller"),
            "agent_policy": agent_decision.get("policy_version", AGENT_OPTIONS.get("policy", "rule_v2")),
            "agent_selected_dataset_agent": (auto_route or {}).get("selected_dataset_agent", "disabled"),
            "agent_route_reason": (auto_route or {}).get("route_reason", "agent_mode_off"),
            "agent_decision": agent_verification.get("final_decision", agent_decision.get("decision")),
            "agent_decision_reason": agent_verification.get("final_decision_reason", agent_decision.get("decision_reason")),
            "agent_verification_status": agent_verification.get("verification_status"),
            "agent_verification_reason": agent_verification.get("verification_reason"),
            "agent_selected_execution_mode": agent_verification.get(
                "selected_execution_mode",
                agent_decision.get("selected_execution_mode"),
            ),
            "agent_stage5_allowed": agent_decision.get("stage5_allowed"),
            "agent_used_stage5": bool(agent_verification.get("use_stage5")),
            "agent_fallback_to_baseline": agent_verification.get("fallback_to_baseline", True),
            "agent_triggered_reverification": bool(agent_verification.get("triggered_reverification")),
            "agent_shadow_used": bool(agent_verification.get("shadow_used")),
            "agent_shadow_accepted": bool(agent_verification.get("shadow_accepted")),
            "agent_signals": agent_decision.get("agent_signals", {}),
            "stage5_applied": bool(agent_verification.get("use_stage5")),
            "stage5_checkpoint": (stage5_prediction or {}).get("checkpoint_path"),
            "stage5_checkpoint_family": (stage5_prediction or {}).get("checkpoint_family", stage5_checkpoint_route.get("task_family")),
            "stage5_checkpoint_source": (stage5_prediction or {}).get("checkpoint_source", stage5_checkpoint_route.get("checkpoint_source")),
            "stage5_checkpoint_route_reason": (stage5_prediction or {}).get("checkpoint_route_reason", stage5_checkpoint_route.get("route_reason")),
            "stage5_checkpoint_route_policy": (stage5_prediction or {}).get("checkpoint_route_policy", stage5_checkpoint_route.get("routing_policy")),
            "stage5_checkpoint_route_signals": (stage5_prediction or {}).get("checkpoint_route_signals", stage5_checkpoint_route.get("signals", {})),
            "stage5_mode": stage5_gate.get("mode"),
            "stage5_skip_reason": (stage5_prediction or {}).get("skip_reason", ""),
            "stage5_raw_direction_attributes": (stage5_prediction or {}).get("raw_direction_attributes", []),
            "stage5_direction_attributes": (stage5_prediction or {}).get("direction_attributes", []),
            "stage5_direction_vector": (stage5_prediction or {}).get("direction_vector"),
            "stage5_target_orientation": (stage5_prediction or {}).get("target_orientation"),
            "stage5_orientation_diagnostics": stage5_orientation_diagnostics,
            "stage5_vector_x": stage5_orientation_diagnostics.get("vector_x"),
            "stage5_vector_y": stage5_orientation_diagnostics.get("vector_y"),
            "stage5_vector_z": stage5_orientation_diagnostics.get("vector_z"),
            "stage5_top_up_score": stage5_orientation_diagnostics.get("top_up_score"),
            "stage5_bottom_down_score": stage5_orientation_diagnostics.get("bottom_down_score"),
            "stage5_semantic_status": stage5_orientation_diagnostics.get("semantic_status"),
            "stage5_semantic_warning": stage5_orientation_diagnostics.get("semantic_warning"),
            "stage5_old_rule_pass": agent_verification.get("old_rule_pass"),
            "stage5_new_rule_pass": agent_verification.get("new_rule_pass"),
            "stage5_target_axis": agent_verification.get("target_axis"),
            "stage5_target_axis_label": agent_verification.get("target_axis_label"),
            "stage5_target_axis_source": agent_verification.get("target_axis_source"),
            "stage5_cosine_to_target_axis": agent_verification.get("cosine_to_target_axis"),
            "stage5_verifier_rule_version": agent_verification.get("verifier_rule_version"),
            "stage5_verifier_decision_reason": agent_verification.get("verifier_decision_reason"),
            "stage5_head": stage5_prediction,
            "stage5_shadow_head": stage5_shadow_prediction,
            "stage5_fallback_scene_graph_used": stage5_fallback_scene_graph_used,
            **stage_times,
        }


def write_csv_records(csv_path, records, fieldnames):
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_stage5_integration_outputs(output_dir, run_id, summary, run_context):
    records = summary.get("records", []) or []
    export_records = []
    for record in records:
        stage5_head = record.get("stage5_head")
        if not stage5_head and not record.get("stage5_enabled"):
            continue
        export_records.append(
            {
                "task_dir": record.get("task_dir"),
                "status": record.get("status"),
                "pipeline_mode": record.get("pipeline_mode"),
                "failed_stage": record.get("failed_stage"),
                "result_file": record.get("result_file"),
                "stage5_enabled": record.get("stage5_enabled"),
                "agent_controller": record.get("agent_controller"),
                "agent_policy": record.get("agent_policy"),
                "agent_selected_dataset_agent": record.get("agent_selected_dataset_agent"),
                "agent_route_reason": record.get("agent_route_reason"),
                "agent_decision": record.get("agent_decision"),
                "agent_decision_reason": record.get("agent_decision_reason"),
                "agent_verification_status": record.get("agent_verification_status"),
                "agent_verification_reason": record.get("agent_verification_reason"),
                "agent_selected_execution_mode": record.get("agent_selected_execution_mode"),
                "agent_stage5_allowed": record.get("agent_stage5_allowed"),
                "agent_used_stage5": record.get("agent_used_stage5"),
                "agent_fallback_to_baseline": record.get("agent_fallback_to_baseline"),
                "agent_triggered_reverification": record.get("agent_triggered_reverification"),
                "agent_shadow_used": record.get("agent_shadow_used"),
                "agent_shadow_accepted": record.get("agent_shadow_accepted"),
                "agent_signals": _json_dumps_safe(record.get("agent_signals", {})),
                "stage5_applied": record.get("stage5_applied"),
                "stage5_checkpoint": record.get("stage5_checkpoint"),
                "stage5_checkpoint_family": record.get("stage5_checkpoint_family"),
                "stage5_checkpoint_source": record.get("stage5_checkpoint_source"),
                "stage5_checkpoint_route_reason": record.get("stage5_checkpoint_route_reason"),
                "stage5_checkpoint_route_policy": record.get("stage5_checkpoint_route_policy"),
                "stage5_checkpoint_route_signals": _json_dumps_safe(record.get("stage5_checkpoint_route_signals", {})),
                "stage5_mode": record.get("stage5_mode"),
                "stage5_skip_reason": record.get("stage5_skip_reason"),
                "stage5_raw_direction_attributes": _json_dumps_safe(record.get("stage5_raw_direction_attributes", [])),
                "stage5_direction_attributes": _json_dumps_safe(record.get("stage5_direction_attributes", [])),
                "stage5_direction_vector": _json_dumps_safe(record.get("stage5_direction_vector")),
                "stage5_target_orientation": _json_dumps_safe(record.get("stage5_target_orientation")),
                "stage5_orientation_diagnostics": _json_dumps_safe(record.get("stage5_orientation_diagnostics", {})),
                "stage5_vector_x": record.get("stage5_vector_x"),
                "stage5_vector_y": record.get("stage5_vector_y"),
                "stage5_vector_z": record.get("stage5_vector_z"),
                "stage5_top_up_score": record.get("stage5_top_up_score"),
                "stage5_bottom_down_score": record.get("stage5_bottom_down_score"),
                "stage5_semantic_status": record.get("stage5_semantic_status"),
                "stage5_semantic_warning": record.get("stage5_semantic_warning"),
                "stage5_old_rule_pass": record.get("stage5_old_rule_pass"),
                "stage5_new_rule_pass": record.get("stage5_new_rule_pass"),
                "stage5_target_axis": _json_dumps_safe(record.get("stage5_target_axis")),
                "stage5_target_axis_label": record.get("stage5_target_axis_label"),
                "stage5_target_axis_source": record.get("stage5_target_axis_source"),
                "stage5_cosine_to_target_axis": record.get("stage5_cosine_to_target_axis"),
                "stage5_verifier_rule_version": record.get("stage5_verifier_rule_version"),
                "stage5_verifier_decision_reason": record.get("stage5_verifier_decision_reason"),
                "stage5_fallback_scene_graph_used": record.get("stage5_fallback_scene_graph_used"),
                "elapsed_sec": record.get("elapsed_sec"),
                "error": record.get("error"),
            }
        )

    if not export_records:
        return None

    export_summary = {
        "mode": "stage5_open6dor_pipeline_integration",
        "run_id": run_id,
        "run_mode": run_context.get("run_mode"),
        "artifact_tag": run_context.get("artifact_tag"),
        "speed_profile": run_context.get("speed_profile"),
        "total_records": len(export_records),
        "applied_count": sum(1 for item in export_records if item.get("stage5_applied")),
        "skipped_count": sum(1 for item in export_records if item.get("stage5_skip_reason")),
        "error_count": sum(1 for item in export_records if item.get("status") == "error"),
        "agent_summary": summarize_open6dor_agent_records(records),
        "records": export_records,
    }
    stable_name = (
        f"stage5_open6dor_pipeline_records_{run_context['artifact_tag']}.json"
        if run_context.get("artifact_tag")
        else "stage5_open6dor_pipeline_records.json"
    )
    stable_path, _ = write_json_outputs(export_summary, output_dir, stable_name, run_id)
    csv_path = stable_path.with_suffix(".csv")
    write_csv_records(
        csv_path,
        export_records,
        [
            "task_dir",
            "status",
            "pipeline_mode",
            "failed_stage",
            "result_file",
            "stage5_enabled",
            "agent_controller",
            "agent_policy",
            "agent_selected_dataset_agent",
            "agent_route_reason",
            "agent_decision",
            "agent_decision_reason",
            "agent_verification_status",
            "agent_verification_reason",
            "agent_selected_execution_mode",
            "agent_stage5_allowed",
            "agent_used_stage5",
            "agent_fallback_to_baseline",
            "agent_triggered_reverification",
            "agent_shadow_used",
            "agent_shadow_accepted",
            "agent_signals",
            "stage5_applied",
            "stage5_checkpoint",
            "stage5_checkpoint_family",
            "stage5_checkpoint_source",
            "stage5_checkpoint_route_reason",
            "stage5_checkpoint_route_policy",
            "stage5_checkpoint_route_signals",
            "stage5_mode",
            "stage5_skip_reason",
            "stage5_raw_direction_attributes",
            "stage5_direction_attributes",
            "stage5_direction_vector",
            "stage5_target_orientation",
            "stage5_orientation_diagnostics",
            "stage5_vector_x",
            "stage5_vector_y",
            "stage5_vector_z",
            "stage5_top_up_score",
            "stage5_bottom_down_score",
            "stage5_semantic_status",
            "stage5_semantic_warning",
            "stage5_old_rule_pass",
            "stage5_new_rule_pass",
            "stage5_target_axis",
            "stage5_target_axis_label",
            "stage5_target_axis_source",
            "stage5_cosine_to_target_axis",
            "stage5_verifier_rule_version",
            "stage5_verifier_decision_reason",
            "stage5_fallback_scene_graph_used",
            "elapsed_sec",
            "error",
        ],
    )
    print(f"[batch-log] wrote {csv_path}")
    return stable_path


def run_stage2_parser_only(dataset_paths, output_dir, run_id, run_context):
    records = []
    for task_dir in tqdm(dataset_paths, total=len(dataset_paths)):
        error = None
        parser_output = None
        prompt = ""
        task_info = {}
        try:
            with open(os.path.join(task_dir, "task_config_new5.json"), "r", encoding="utf-8") as f:
                task_info = json.load(f)
            prompt = (
                f"Pick the {task_info['target_obj_name']}. " + task_info["instruction"]
                if "rot" in task_dir
                else task_info["instruction"]
            )
            image_path = os.path.join(task_dir, "isaac_render-rgb-0-1.png")
            image = Image.open(image_path).convert("RGB")
            parser_output = open6dor_stage2_fast_parser_fn(prompt, image, task_info)
        except Exception as exc:
            error = str(exc)
            print(f"[stage2-parser][open6dor] {task_dir} failed: {exc}")
        finally:
            try:
                del image
            except Exception:
                pass

        records.append(
            {
                "task_dir": task_dir,
                "picked_object": (parser_output or {}).get("picked_object", ""),
                "related_objects": _json_dumps_safe((parser_output or {}).get("related_objects", [])),
                "orientation_mode": (parser_output or {}).get("orientation_mode", ""),
                "relation": (parser_output or {}).get("relation", ""),
                "reference_frame": (parser_output or {}).get("reference_frame", ""),
                "direction_attributes": _json_dumps_safe((parser_output or {}).get("direction_attributes", [])),
                "parser_confidence": (parser_output or {}).get("parser_confidence"),
                "routing_hints": _json_dumps_safe((parser_output or {}).get("routing_hints", {})),
                "error": error,
                "instruction": prompt,
                "parser_output": parser_output,
            }
        )

    summary = {
        "mode": "stage2_parser_only",
        "dataset": "open6dor",
        "pilot": run_context.get("pilot"),
        "task_list": run_context.get("task_list"),
        "total_tasks": len(records),
        "success_count": sum(1 for item in records if not item["error"]),
        "error_count": sum(1 for item in records if item["error"]),
        "records": records,
    }
    stable_name = (
        f"stage2_open6dor_parser_records_{run_context['artifact_tag']}.json"
        if run_context.get("artifact_tag")
        else "stage2_open6dor_parser_records.json"
    )
    stable_path, _ = write_json_outputs(summary, output_dir, stable_name, run_id)
    csv_path = stable_path.with_suffix(".csv")
    write_csv_records(
        csv_path,
        records,
        [
            "task_dir",
            "instruction",
            "picked_object",
            "related_objects",
            "orientation_mode",
            "relation",
            "reference_frame",
            "direction_attributes",
            "parser_confidence",
            "routing_hints",
            "error",
        ],
    )
    print(f"[batch-log] wrote {csv_path}")
    return summary


def _stage3_fast_part_query(parser_output):
    queries = _stage3_fast_part_queries(parser_output)
    return queries[0] if queries else ""


def _dedupe_stage3_queries(values):
    deduped = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        deduped.append(text)
        seen.add(key)
    return deduped


def _stage3_object_queries(picked_object):
    queries = [normalize_object_name(picked_object)]
    template = resolve_orientation_template(picked_object, ORIENTATION_TEMPLATES)
    if template:
        queries.extend(template.get("aliases", []))
    alias_fallbacks = {
        "usb": ["flash drive", "u disk"],
        "mobile phone": ["phone", "smartphone", "cell phone"],
        "binder clips": ["binder clip", "clips"],
    }
    canonical = normalize_object_name(picked_object).lower()
    queries.extend(alias_fallbacks.get(canonical, []))
    return _dedupe_stage3_queries(queries)


def _stage3_fast_part_queries(parser_output):
    direction_attributes = [
        str(value).strip()
        for value in parser_output.get("direction_attributes", [])
        if str(value or "").strip()
    ]
    orientation_mode = str(parser_output.get("orientation_mode", "")).strip().lower()
    picked_object = parser_output.get("picked_object", "")
    template_hints = build_orientation_template_hints(picked_object, ORIENTATION_TEMPLATES)
    template_attrs = template_hints.get("direction_attributes", [])

    preferred = []
    if orientation_mode.startswith("plug_"):
        preferred = ["silver plug end", "plug end", "larger end"]
    elif orientation_mode.startswith("handle_"):
        preferred = ["handle"]
    elif orientation_mode in {"upright", "upside_down"}:
        preferred = ["top", "bottom", "cap", "opening"]
    elif orientation_mode == "lying_flat":
        preferred = ["larger face", "front cover", "screen", "back"]
    elif orientation_mode in {"lying_sideways", "clip_sideways"}:
        preferred = ["clip side", "spine", "handle", "side"]

    queries = []
    queries.extend(preferred)
    queries.extend(direction_attributes)
    queries.extend(template_attrs)
    return _dedupe_stage3_queries(queries)


def _run_detection_with_queries(image, queries, output_folder):
    last_detections = None
    last_score = None
    for query in _dedupe_stage3_queries(queries):
        detections = detection.get_detections(
            image,
            [query],
            detection_model,
            output_folder=output_folder,
            single=True,
            save_artifacts=False,
        )
        bbox = first_detection_xyxy(detections, image.width, image.height)
        score = first_detection_score(detections)
        last_detections = detections
        last_score = score
        if bbox is not None:
            return query, detections, bbox, score
    return "", last_detections, None, last_score


def run_stage3_grounding_only(dataset_paths, output_dir, run_id, run_context):
    records = []
    for task_dir in tqdm(dataset_paths, total=len(dataset_paths)):
        start_time = time.perf_counter()
        error = None
        failed_stage = None
        status = "success"
        parser_output = None
        object_bbox = None
        part_bbox = None
        object_score = None
        part_score = None
        object_mask = None
        part_mask = None
        cache_paths = {}
        prompt = ""
        task_info = {}
        part_query = ""
        timings = {
            "parser_sec": 0.0,
            "object_grounding_sec": 0.0,
            "object_sam_sec": 0.0,
            "part_grounding_sec": 0.0,
            "part_sam_sec": 0.0,
        }

        try:
            image_path = os.path.join(task_dir, "isaac_render-rgb-0-1.png")
            with open(os.path.join(task_dir, "task_config_new5.json"), "r", encoding="utf-8") as f:
                task_info = json.load(f)
            prompt = (
                f"Pick the {task_info['target_obj_name']}. " + task_info["instruction"]
                if "rot" in task_dir
                else task_info["instruction"]
            )
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            stage_start = time.perf_counter()
            parser_output = open6dor_stage2_fast_parser_fn(prompt, image, task_info)
            timings["parser_sec"] = round(time.perf_counter() - stage_start, 2)

            picked_object = str(parser_output.get("picked_object", "")).strip()
            if not picked_object:
                failed_stage = "parser"
                raise ValueError("Stage 3 requires a non-empty picked_object from the fast parser")
            part_queries = _stage3_fast_part_queries(parser_output)
            part_query = part_queries[0] if part_queries else ""

            stage_start = time.perf_counter()
            _, object_detections, object_bbox, object_score = _run_detection_with_queries(
                image,
                _stage3_object_queries(picked_object),
                output_dir,
            )
            timings["object_grounding_sec"] = round(time.perf_counter() - stage_start, 2)
            if object_bbox is None:
                failed_stage = "object_grounding"
                raise ValueError("Object grounding returned no usable bbox")

            stage_start = time.perf_counter()
            object_masks, _, _ = sam.get_mask(
                image,
                [picked_object],
                sam_model,
                object_detections,
                output_folder=output_dir,
                save_artifacts=False,
                prefer_single_mask=True,
            )
            timings["object_sam_sec"] = round(time.perf_counter() - stage_start, 2)
            if len(object_masks) == 0:
                failed_stage = "object_sam"
                raise ValueError("Object SAM returned no mask")
            object_mask = object_masks[0]

            if part_queries:
                roi_image_np = crop_image_array(image_np, object_bbox)
                roi_image = Image.fromarray(roi_image_np)
                stage_start = time.perf_counter()
                part_query, part_detections, roi_part_bbox, part_score = _run_detection_with_queries(
                    roi_image,
                    part_queries,
                    output_dir,
                )
                timings["part_grounding_sec"] = round(time.perf_counter() - stage_start, 2)
                if roi_part_bbox is None:
                    status = "partial"
                    failed_stage = "part_grounding"
                else:
                    stage_start = time.perf_counter()
                    part_masks, _, _ = sam.get_mask(
                        roi_image,
                        [part_query],
                        sam_model,
                        part_detections,
                        output_folder=output_dir,
                        save_artifacts=False,
                        prefer_single_mask=True,
                    )
                    timings["part_sam_sec"] = round(time.perf_counter() - stage_start, 2)
                    if len(part_masks) == 0:
                        status = "partial"
                        failed_stage = "part_sam"
                    else:
                        part_mask = expand_roi_mask(part_masks[0], object_mask.shape, object_bbox)
                        part_mask = constrain_child_mask_to_parent(part_mask, object_mask)
                        part_bbox = offset_xyxy(roi_part_bbox, object_bbox, image.width, image.height)

            stage3_dir = Path(task_dir) / "output" / "stage3"
            cache_payload = {
                "dataset": "open6dor",
                "task_dir": task_dir,
                "instruction": prompt,
                "parser_output": parser_output,
                "grounding": {
                    "picked_object": parser_output.get("picked_object", ""),
                    "related_objects": parser_output.get("related_objects", []),
                    "orientation_mode": parser_output.get("orientation_mode", ""),
                    "reference_frame": parser_output.get("reference_frame", ""),
                    "part_query": part_query,
                    "object_bbox_xyxy": serialize_xyxy(object_bbox),
                    "part_bbox_xyxy": serialize_xyxy(part_bbox),
                    "object_score": object_score,
                    "part_score": part_score,
                    "reference_strategy": "lightweight_only",
                    "image_size": [image.width, image.height],
                    "status": status,
                    "failed_stage": failed_stage,
                    "timings": timings,
                },
            }
            cache_paths = save_stage3_cache(stage3_dir, cache_payload, object_mask, part_mask)
        except Exception as exc:
            error = str(exc)
            status = "error"
            print(f"[stage3-grounding][open6dor] {task_dir} failed: {exc}")
            stage3_dir = Path(task_dir) / "output" / "stage3"
            cache_payload = {
                "dataset": "open6dor",
                "task_dir": task_dir,
                "instruction": prompt,
                "parser_output": parser_output,
                "grounding": {
                    "picked_object": (parser_output or {}).get("picked_object", ""),
                    "related_objects": (parser_output or {}).get("related_objects", []),
                    "orientation_mode": (parser_output or {}).get("orientation_mode", ""),
                    "reference_frame": (parser_output or {}).get("reference_frame", ""),
                    "part_query": part_query,
                    "object_bbox_xyxy": serialize_xyxy(object_bbox),
                    "part_bbox_xyxy": serialize_xyxy(part_bbox),
                    "object_score": object_score,
                    "part_score": part_score,
                    "reference_strategy": "lightweight_only",
                    "status": status,
                    "failed_stage": failed_stage,
                    "timings": timings,
                    "error": error,
                },
            }
            cache_paths = save_stage3_cache(stage3_dir, cache_payload, object_mask, part_mask)

        total_sec = round(time.perf_counter() - start_time, 2)
        records.append(
            {
                "task_dir": task_dir,
                "picked_object": (parser_output or {}).get("picked_object", ""),
                "related_objects": _json_dumps_safe((parser_output or {}).get("related_objects", [])),
                "orientation_mode": (parser_output or {}).get("orientation_mode", ""),
                "reference_frame": (parser_output or {}).get("reference_frame", ""),
                "part_query": part_query,
                "status": status,
                "failed_stage": failed_stage,
                "object_bbox_xyxy": _json_dumps_safe(serialize_xyxy(object_bbox)),
                "part_bbox_xyxy": _json_dumps_safe(serialize_xyxy(part_bbox)),
                "object_score": object_score,
                "part_score": part_score,
                "roi_meta_path": cache_paths.get("roi_meta_path"),
                "cache_path": cache_paths.get("cache_path"),
                "object_mask_path": cache_paths.get("object_mask_path"),
                "part_mask_path": cache_paths.get("part_mask_path"),
                "parser_sec": timings["parser_sec"],
                "object_grounding_sec": timings["object_grounding_sec"],
                "object_sam_sec": timings["object_sam_sec"],
                "part_grounding_sec": timings["part_grounding_sec"],
                "part_sam_sec": timings["part_sam_sec"],
                "total_sec": total_sec,
                "error": error,
            }
        )

    summary = {
        "mode": "stage3_grounding_only",
        "dataset": "open6dor",
        "pilot": run_context.get("pilot"),
        "task_list": run_context.get("task_list"),
        "total_tasks": len(records),
        "success_count": sum(1 for item in records if item["status"] == "success"),
        "partial_count": sum(1 for item in records if item["status"] == "partial"),
        "error_count": sum(1 for item in records if item["status"] == "error"),
        "records": records,
    }
    stable_name = (
        f"stage3_open6dor_grounding_records_{run_context['artifact_tag']}.json"
        if run_context.get("artifact_tag")
        else "stage3_open6dor_grounding_records.json"
    )
    stable_path, _ = write_json_outputs(summary, output_dir, stable_name, run_id)
    csv_path = stable_path.with_suffix(".csv")
    write_csv_records(
        csv_path,
        records,
        [
            "task_dir",
            "picked_object",
            "related_objects",
            "orientation_mode",
            "reference_frame",
            "part_query",
            "status",
            "failed_stage",
            "object_bbox_xyxy",
            "part_bbox_xyxy",
            "object_score",
            "part_score",
            "roi_meta_path",
            "cache_path",
            "object_mask_path",
            "part_mask_path",
            "parser_sec",
            "object_grounding_sec",
            "object_sam_sec",
            "part_grounding_sec",
            "part_sam_sec",
            "total_sec",
            "error",
        ],
    )
    print(f"[batch-log] wrote {csv_path}")
    return summary


def run_stage4_pointdata_only(dataset_paths, output_dir, run_id, run_context):
    records = []
    sample_points_num = int(os.getenv("SOFAR_STAGE4_SAMPLE_POINTS", "4096"))
    for task_dir in tqdm(dataset_paths, total=len(dataset_paths)):
        start_time = time.perf_counter()
        error = None
        status = "success"
        object_points = np.zeros((0, 6), dtype=np.float32)
        part_points = np.zeros((0, 6), dtype=np.float32)
        priors = {}
        cache_paths = {}
        try:
            image_path = os.path.join(task_dir, "isaac_render-rgb-0-1.png")
            depth_path = os.path.join(task_dir, "isaac_render-depth-0-1.npy")
            stage3_dir = Path(task_dir) / "output" / "stage3"
            cache_json = stage3_dir / "object_part_cache.json"
            object_mask_path = stage3_dir / "object_mask.npz"
            part_mask_path = stage3_dir / "part_mask.npz"
            if not cache_json.exists() or not object_mask_path.exists():
                raise FileNotFoundError("Stage 4 requires existing Stage 3 cache files")

            with cache_json.open("r", encoding="utf-8") as f:
                stage3_cache = json.load(f)
            object_mask = np.load(object_mask_path)["mask"]
            part_mask = np.load(part_mask_path)["mask"] if part_mask_path.exists() else None
            part_mask = constrain_child_mask_to_parent(part_mask, object_mask)

            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            depth = np.load(depth_path)
            vinvs = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-0.9028605, -0.0, 0.42993355, -0.0],
                    [0.42993355, -0.0, 0.9028605, -0.0],
                    [1.0, 0.0, 1.2, 1.0],
                ]
            )
            projs = [
                [1.7320507, 0.0, 0.0, 0.0],
                [0.0, 2.5980759, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.05, 0.0],
            ]
            pcd = get_point_cloud_from_rgbd(depth, np.array(image), vinvs, projs).cpu().numpy().astype(np.float32)
            pcd = pcd.reshape(depth.shape[0], depth.shape[1], 6)

            object_points = sample_points(pcd[object_mask], sample_points_num)
            part_points = (
                sample_points(pcd[part_mask], sample_points_num)
                if part_mask is not None and part_mask.any()
                else np.zeros((0, 6), dtype=np.float32)
            )
            priors = compute_geometry_priors(object_points, part_points)
            stage4_payload = {
                "dataset": "open6dor",
                "task_dir": task_dir,
                "stage3_cache_path": str(cache_json),
                "parser_output": stage3_cache.get("parser_output", {}),
                "grounding": stage3_cache.get("grounding", {}),
                "geometry_priors": priors,
                "sample_points": sample_points_num,
            }
            cache_paths = save_stage4_cache(Path(task_dir) / "output" / "stage4", stage4_payload, object_points, part_points)
        except Exception as exc:
            error = str(exc)
            status = "error"
            print(f"[stage4-pointdata][open6dor] {task_dir} failed: {exc}")

        total_sec = round(time.perf_counter() - start_time, 2)
        records.append(
            {
                "task_dir": task_dir,
                "status": status,
                "object_point_count": int(len(object_points)),
                "part_point_count": int(len(part_points)),
                "part_ratio": priors.get("part_ratio"),
                "cache_path": cache_paths.get("cache_path"),
                "object_points_path": cache_paths.get("object_points_path"),
                "part_points_path": cache_paths.get("part_points_path"),
                "total_sec": total_sec,
                "error": error,
            }
        )

    summary = {
        "mode": "stage4_pointdata_only",
        "dataset": "open6dor",
        "pilot": run_context.get("pilot"),
        "task_list": run_context.get("task_list"),
        "total_tasks": len(records),
        "success_count": sum(1 for item in records if item["status"] == "success"),
        "error_count": sum(1 for item in records if item["status"] == "error"),
        "records": records,
    }
    stable_name = (
        f"stage4_open6dor_point_records_{run_context['artifact_tag']}.json"
        if run_context.get("artifact_tag")
        else "stage4_open6dor_point_records.json"
    )
    stable_path, _ = write_json_outputs(summary, output_dir, stable_name, run_id)
    csv_path = stable_path.with_suffix(".csv")
    write_csv_records(
        csv_path,
        records,
        [
            "task_dir",
            "status",
            "object_point_count",
            "part_point_count",
            "part_ratio",
            "cache_path",
            "object_points_path",
            "part_points_path",
            "total_sec",
            "error",
        ],
    )
    print(f"[batch-log] wrote {csv_path}")
    return summary


if __name__ == "__main__":
    args = parse_args()
    selected_modes = sum(
        [
            args.stage2_parser_only,
            args.stage3_grounding_only,
            args.stage4_pointdata_only,
        ]
    )
    if selected_modes > 1:
        raise ValueError("Only one of --stage2-parser-only, --stage3-grounding-only, or --stage4-pointdata-only may be used at a time.")
    output_dir = runtime_paths.ensure_output_dir()
    dataset_root = runtime_paths.open6dor_dataset_dir()
    run_context = build_run_context(args, dataset_root)
    run_context["speed_profile_settings"] = apply_speed_profile(args.speed_profile)

    log_prefix = run_context["log_prefix"]
    if args.stage2_parser_only:
        log_prefix = f"stage2_{log_prefix}"
    elif args.stage3_grounding_only:
        log_prefix = f"stage3_{log_prefix}"
    elif args.stage4_pointdata_only:
        log_prefix = f"stage4_{log_prefix}"
    run_id, _ = setup_timestamped_logging(output_dir, log_prefix)
    dataset_paths = run_context["dataset_paths"]
    RUN_RECORDS = []
    processed_task_dirs = set()
    stage_only_mode = args.stage2_parser_only or args.stage3_grounding_only or args.stage4_pointdata_only
    if not stage_only_mode:
        RUN_RECORDS, processed_task_dirs = load_progress(
            output_dir,
            run_context["artifact_tag"],
            reset_progress=args.reset_progress,
        )
        if processed_task_dirs:
            print(f"[open6dor] resuming from progress file with {len(processed_task_dirs)} completed tasks")
        if args.reset_progress:
            print("[open6dor] starting fresh because --reset-progress was provided")

    print(f"[open6dor] run_mode={run_context['run_mode']}")
    if run_context.get("pilot"):
        print(f"[open6dor] pilot={run_context['pilot']}")
    if run_context.get("task_list"):
        print(f"[open6dor] task_list={run_context['task_list']}")
    print(f"[open6dor] speed_profile={run_context['speed_profile']}")
    if run_context["speed_profile_settings"]:
        print(f"[open6dor] speed_profile_settings={run_context['speed_profile_settings']}")
    print(f"[open6dor] discovered {len(dataset_paths)} task directories")
    llm_backend = resolve_llm_backend()
    print(f"LLM backend: {llm_backend}")

    if llm_backend == "qwen" and not args.stage4_pointdata_only:
        from serve.qwen_inference import stage2_fast_open6dor_parser as qwen_stage2_fast_open6dor_parser
        from serve.qwen_inference import open6dor_joint_reasoning as qwen_open6dor_joint_reasoning
        from serve.qwen_inference import open6dor_parsing as qwen_open6dor_parsing
        from serve.qwen_inference import open6dor_spatial_reasoning as qwen_open6dor_spatial_reasoning

        qwen_model, processor = load_qwen_model()

        def open6dor_joint_reasoning_fn(img, command, scene_graph, object_set_hint, orientation_template_hints):
            return qwen_open6dor_joint_reasoning(
                qwen_model,
                processor,
                img,
                command,
                scene_graph,
                object_set_hint=object_set_hint,
                orientation_template_hints=orientation_template_hints,
            )

        def open6dor_parsing_fn(command, img):
            return qwen_open6dor_parsing(qwen_model, processor, img, command)

        def open6dor_spatial_reasoning_fn(img, command, picked_info, other_info):
            return qwen_open6dor_spatial_reasoning(qwen_model, processor, img, command, picked_info, other_info)

        def open6dor_stage2_fast_parser_fn(command, img, task_info):
            return qwen_stage2_fast_open6dor_parser(qwen_model, processor, img, command, task_info)

    elif llm_backend != "qwen":
        from serve.chatgpt import open6dor_parsing as chatgpt_open6dor_parsing
        from serve.chatgpt import open6dor_spatial_reasoning as chatgpt_open6dor_spatial_reasoning

        open6dor_joint_reasoning_fn = None

        def open6dor_parsing_fn(command, img):
            return chatgpt_open6dor_parsing(command, img)

        def open6dor_spatial_reasoning_fn(img, command, picked_info, other_info):
            return chatgpt_open6dor_spatial_reasoning(img, command, picked_info, other_info)

        def open6dor_stage2_fast_parser_fn(command, img, task_info):
            raise NotImplementedError("Stage 2 fast parser smoke is currently implemented for the qwen backend only.")

    if args.stage2_parser_only:
        if args.limit is not None:
            dataset_paths = dataset_paths[: args.limit]
            print(f"[open6dor] applying --limit {args.limit}, parser smoke reduced to {len(dataset_paths)} tasks")
        run_stage2_parser_only(dataset_paths, output_dir, run_id, run_context)
        raise SystemExit(0)
    if args.stage3_grounding_only:
        if llm_backend != "qwen":
            raise NotImplementedError("Stage 3 grounding-only mode currently requires the qwen backend for Stage 2 fast parser output.")
        detection_model = detection.get_model()
        sam_model = sam.get_model()
        RUN_OPTIONS.update(
            {
                "save_debug_artifacts": _env_flag("SOFAR_SAVE_DEBUG_ARTIFACTS", False),
                "prefer_single_mask": _env_flag("SOFAR_SAM_PREFER_SINGLE_MASK", True),
                "use_perception_cache": False,
            }
        )
        print(f"[open6dor] save_debug_artifacts={RUN_OPTIONS['save_debug_artifacts']}")
        print(f"[open6dor] prefer_single_mask={RUN_OPTIONS['prefer_single_mask']}")
        dataset_paths, skipped_count = filter_pending_stage_cache_tasks(dataset_paths, "stage3")
        if skipped_count:
            print(f"[open6dor] skipped {skipped_count} tasks with existing Stage 3 cache")
        if args.limit is not None:
            dataset_paths = dataset_paths[: args.limit]
            print(f"[open6dor] applying --limit {args.limit}, stage3 grounding reduced to {len(dataset_paths)} pending tasks")
        run_stage3_grounding_only(dataset_paths, output_dir, run_id, run_context)
        raise SystemExit(0)
    if args.stage4_pointdata_only:
        dataset_paths, skipped_count = filter_pending_stage_cache_tasks(dataset_paths, "stage4")
        if skipped_count:
            print(f"[open6dor] skipped {skipped_count} tasks with existing Stage 4 cache")
        if args.limit is not None:
            dataset_paths = dataset_paths[: args.limit]
            print(f"[open6dor] applying --limit {args.limit}, stage4 point-data reduced to {len(dataset_paths)} pending tasks")
        run_stage4_pointdata_only(dataset_paths, output_dir, run_id, run_context)
        raise SystemExit(0)

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()
    ORIENTATION_TEMPLATES = load_orientation_templates()
    stage5_mode_spec = args.stage5_open6dor_modes or os.getenv("SOFAR_STAGE5_OPEN6DOR_MODES", "")
    allowed_stage5_modes = _parse_stage5_mode_list(
        stage5_mode_spec,
        default_modes=DEFAULT_STAGE5_OPEN6DOR_ALLOWED_MODES,
    )

    RUN_OPTIONS.update(
        {
            "save_debug_artifacts": _env_flag("SOFAR_SAVE_DEBUG_ARTIFACTS", True),
            "prefer_single_mask": _env_flag("SOFAR_SAM_PREFER_SINGLE_MASK", False),
            "use_perception_cache": _env_flag("SOFAR_OPEN6DOR_USE_PERCEPTION_CACHE", True),
        }
    )
    print(f"[open6dor] save_debug_artifacts={RUN_OPTIONS['save_debug_artifacts']}")
    print(f"[open6dor] prefer_single_mask={RUN_OPTIONS['prefer_single_mask']}")
    print(f"[open6dor] use_perception_cache={RUN_OPTIONS['use_perception_cache']}")
    shared_stage5_checkpoint = _optional_path_string(args.stage5_checkpoint)
    if not shared_stage5_checkpoint:
        shared_stage5_checkpoint = str(runtime_paths.stage5_open6dor_checkpoint_path())
    upright_expert_checkpoint = _optional_path_string(args.stage5_upright_expert_checkpoint)
    if not upright_expert_checkpoint:
        upright_expert_checkpoint = _optional_path_string(runtime_paths.stage5_open6dor_upright_checkpoint_path())
    if not upright_expert_checkpoint:
        upright_expert_checkpoint = shared_stage5_checkpoint
    flat_expert_checkpoint = _optional_path_string(args.stage5_flat_expert_checkpoint)
    if not flat_expert_checkpoint:
        flat_expert_checkpoint = _optional_path_string(runtime_paths.stage5_open6dor_flat_checkpoint_path())
    plug_expert_checkpoint = _optional_path_string(args.stage5_plug_expert_checkpoint)
    if not plug_expert_checkpoint:
        plug_expert_checkpoint = _optional_path_string(runtime_paths.stage5_open6dor_plug_checkpoint_path())
    STAGE5_OPTIONS.clear()
    STAGE5_OPTIONS.update(
        {
            "enabled": bool(args.use_stage5_head),
            "checkpoint_path": shared_stage5_checkpoint,
            "device": args.stage5_device,
            "num_points": args.stage5_num_points,
            "allowed_modes": allowed_stage5_modes,
            "expert_routing": args.stage5_expert_routing,
            "expert_checkpoints": {
                STAGE5_TASK_FAMILY_UPRIGHT: upright_expert_checkpoint,
                STAGE5_TASK_FAMILY_FLAT: flat_expert_checkpoint,
                STAGE5_TASK_FAMILY_PLUG: plug_expert_checkpoint,
            },
        }
    )
    AGENT_OPTIONS.clear()
    AGENT_OPTIONS.update(
        {
            "mode": args.agent_mode or ("dataset" if args.use_stage5_head else "off"),
            "policy": args.agent_policy,
            "save_trace": bool(args.agent_save_trace),
            "debug_dir": args.agent_debug_dir,
            "shadow_eval": bool(args.agent_shadow_eval),
            "extended_actions": bool(args.agent_extended_actions),
        }
    )
    if STAGE5_OPTIONS["enabled"]:
        print(f"[open6dor] stage5_head=enabled")
        if STAGE5_OPTIONS["checkpoint_path"]:
            print(f"[open6dor] stage5_checkpoint={STAGE5_OPTIONS['checkpoint_path']}")
        print(f"[open6dor] stage5_device={STAGE5_OPTIONS['device']}")
        print(f"[open6dor] stage5_num_points={STAGE5_OPTIONS['num_points']}")
        print(f"[open6dor] stage5_expert_routing={STAGE5_OPTIONS['expert_routing']}")
        if STAGE5_OPTIONS["expert_routing"] != "off":
            for family_name, checkpoint_path in sorted(STAGE5_OPTIONS["expert_checkpoints"].items()):
                if checkpoint_path:
                    print(f"[open6dor] stage5_expert_{family_name}={checkpoint_path}")
                else:
                    print(f"[open6dor] stage5_expert_{family_name}=unset")
        if "all" in STAGE5_OPTIONS["allowed_modes"]:
            print("[open6dor] stage5_open6dor_modes=all")
        else:
            print("[open6dor] stage5_open6dor_modes=" + ",".join(sorted(STAGE5_OPTIONS["allowed_modes"])))
    print(f"[open6dor] agent_mode={AGENT_OPTIONS['mode']}")
    print(f"[open6dor] agent_policy={AGENT_OPTIONS['policy']}")
    if AGENT_OPTIONS["save_trace"]:
        print("[open6dor] agent_trace=enabled")
    if AGENT_OPTIONS["shadow_eval"]:
        print("[open6dor] agent_shadow_eval=enabled")
    if AGENT_OPTIONS["extended_actions"]:
        print("[open6dor] agent_extended_actions=requested (no-op in v1)")

    pending_dataset_paths = [p for p in dataset_paths if p not in processed_task_dirs]
    skipped_existing = []
    runnable_dataset_paths = []
    for task_dir in pending_dataset_paths:
        if should_skip_existing_result(task_dir, rerun_existing=run_context["rerun_existing"]):
            skipped_existing.append(
                {
                    "task_dir": task_dir,
                    "status": "skipped",
                    "result_file": os.path.join(task_dir, "output", "result.json"),
                    "elapsed_sec": 0.0,
                    "total_sec": 0.0,
                }
            )
        else:
            runnable_dataset_paths.append(task_dir)
    if skipped_existing:
        RUN_RECORDS.extend(skipped_existing)
        save_progress(output_dir, run_id, dataset_root, dataset_paths, run_context)
        print(f"[open6dor] skipped {len(skipped_existing)} tasks with existing result.json")
    if args.limit is not None:
        runnable_dataset_paths = runnable_dataset_paths[: args.limit]
        print(f"[open6dor] applying --limit {args.limit}, runnable tasks reduced to {len(runnable_dataset_paths)}")
    print(f"[open6dor] pending runnable tasks: {len(runnable_dataset_paths)}")

    if llm_backend == "qwen":
        for task_dir in tqdm(runnable_dataset_paths):
            record = process_dataset(task_dir)
            RUN_RECORDS.append(record)
            save_progress(output_dir, run_id, dataset_root, dataset_paths, run_context)
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for record in tqdm(executor.map(process_dataset, runnable_dataset_paths), total=len(runnable_dataset_paths)):
                RUN_RECORDS.append(record)
                save_progress(output_dir, run_id, dataset_root, dataset_paths, run_context)

    summary = save_progress(output_dir, run_id, dataset_root, dataset_paths, run_context)
    write_json_outputs(summary, output_dir, run_context["summary_name"], run_id)
    if STAGE5_OPTIONS.get("enabled"):
        write_stage5_integration_outputs(output_dir, run_id, summary, run_context)
        if AGENT_OPTIONS.get("save_trace") or AGENT_OPTIONS.get("mode") in {"dataset", "auto"}:
            debug_dir = write_agent_trace_bundle(
                records=summary.get("records", []),
                output_dir=output_dir,
                dataset="open6dor",
                run_id=run_id,
                debug_root=AGENT_OPTIONS.get("debug_dir"),
            )
            print(f"[open6dor] agent_debug_dir={debug_dir}")
