import argparse
import json
import os
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
)
from serve.scene_graph import build_open6dor_lightweight_scene_graph, open6dor_scene_graph
from segmentation import sam, florence as detection
from serve.utils import generate_rotation_matrix, get_point_cloud_from_rgbd
from serve import runtime_paths
from serve.batch_logging import setup_timestamped_logging, write_json_outputs
from serve.qwen_runtime import resolve_qwen_dtype

warnings.filterwarnings("ignore")
open6dor_parsing_fn = None
open6dor_joint_reasoning_fn = None
open6dor_spatial_reasoning_fn = None
detection_model = None
sam_model = None
orientation_model = None
RUN_RECORDS = []
RUN_OPTIONS = {}
ORIENTATION_TEMPLATES = {}
DEFAULT_PROGRESS_FILE = "open6dor_perception_progress.json"
DEFAULT_SUMMARY_FILE = "open6dor_perception_summary.json"
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


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        json.dump(summary, f, indent=2, ensure_ascii=False)
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
        rerun_existing = False

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


def process_dataset(task_dir):
    start_time = time.perf_counter()
    stage_times = {}
    failed_stage = None
    perception_cache_hit = False
    pipeline_mode = "legacy"

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
                lambda: open6dor_scene_graph(
                    preprocessed_image,
                    pcd,
                    mask,
                    parsed_info,
                    object_names,
                    orientation_model,
                    output_folder=output_path,
                    save_debug_artifacts=RUN_OPTIONS["save_debug_artifacts"],
                ),
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
                        lambda: open6dor_scene_graph(
                            preprocessed_image,
                            pcd,
                            mask,
                            parsed_info_for_perception,
                            object_names,
                            orientation_model,
                            output_folder=output_path,
                            save_debug_artifacts=RUN_OPTIONS["save_debug_artifacts"],
                        ),
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

        init_position = picked_object_dict["center"]
        init_orientation = picked_object_dict["orientation"]
        target_orientation = parsed_info["target_orientation"]

        if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
            direction_attributes = target_orientation.keys()
            init_directions = [init_orientation[direction] for direction in direction_attributes]
            target_directions = [target_orientation[direction] for direction in direction_attributes]
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
        }
        with open(os.path.join(output_path, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print("Successfully saved result for", output_path)
        elapsed = round(time.perf_counter() - start_time, 2)
        return {
            "task_dir": task_dir,
            "status": "success",
            "result_file": os.path.join(output_path, "result.json"),
            "elapsed_sec": elapsed,
            "total_sec": elapsed,
            "perception_cache_hit": perception_cache_hit,
            "pipeline_mode": pipeline_mode,
            **stage_times,
        }

    except Exception as exc:
        elapsed = round(time.perf_counter() - start_time, 2)
        print(f"Error processing {task_dir}: {exc}")
        return {
            "task_dir": task_dir,
            "status": "error",
            "error": str(exc),
            "elapsed_sec": elapsed,
            "total_sec": elapsed,
            "failed_stage": failed_stage,
            "perception_cache_hit": perception_cache_hit,
            "pipeline_mode": pipeline_mode,
            **stage_times,
        }


if __name__ == "__main__":
    args = parse_args()
    output_dir = runtime_paths.ensure_output_dir()
    dataset_root = runtime_paths.open6dor_dataset_dir()
    run_context = build_run_context(args, dataset_root)
    run_context["speed_profile_settings"] = apply_speed_profile(args.speed_profile)

    run_id, _ = setup_timestamped_logging(output_dir, run_context["log_prefix"])
    dataset_paths = run_context["dataset_paths"]
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

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()
    ORIENTATION_TEMPLATES = load_orientation_templates()
    llm_backend = resolve_llm_backend()
    print(f"LLM backend: {llm_backend}")

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

    if llm_backend == "qwen":
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

    else:
        from serve.chatgpt import open6dor_parsing as chatgpt_open6dor_parsing
        from serve.chatgpt import open6dor_spatial_reasoning as chatgpt_open6dor_spatial_reasoning

        open6dor_joint_reasoning_fn = None

        def open6dor_parsing_fn(command, img):
            return chatgpt_open6dor_parsing(command, img)

        def open6dor_spatial_reasoning_fn(img, command, picked_info, other_info):
            return chatgpt_open6dor_spatial_reasoning(img, command, picked_info, other_info)

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
