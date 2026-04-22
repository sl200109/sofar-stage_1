import os
import argparse
import json
import csv
import warnings
import sys
import gc
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve import pointso as orientation
from serve.scene_graph import get_scene_graph
from depth import metric3dv2 as depth_esti_model
from segmentation import sam, florence as detection
from concurrent.futures import ThreadPoolExecutor, as_completed
from serve import runtime_paths
from serve.batch_logging import setup_timestamped_logging, write_json_outputs
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
from serve.stage5_inference import inject_prediction_into_scene_graph, predict_from_stage4_dir
from serve.spatialbench_stage5 import (
    build_spatialbench_stage5_context,
    classify_spatialbench_stage5_applicability,
)

warnings.filterwarnings("ignore")
output_folder = str(runtime_paths.ensure_output_dir())
spatialbench_dir = runtime_paths.spatialbench_dataset_dir()
parse_fn = None
reason_fn = None
PROGRESS_FILE = "eval_spatialbench_progress.json"
STAGE5_OPTIONS = {}


def resolve_llm_backend():
    backend = os.getenv("SOFAR_LLM_BACKEND", "qwen").strip().lower()
    if backend not in {"openai", "qwen"}:
        raise ValueError(f"Unsupported SOFAR_LLM_BACKEND: {backend}")
    return backend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recover-log",
        type=str,
        default=None,
        help="Recover completed samples from an old eval_spatialbench log file.",
    )
    parser.add_argument(
        "--speed-profile",
        choices=["off", "conservative"],
        default="off",
        help="Apply a named batch speed profile for recurring validation runs.",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Ignore eval_spatialbench_progress.json and rerun the full dataset.",
    )
    parser.add_argument(
        "--stage2-parser-only",
        action="store_true",
        help="Run Stage 2 parser smoke only and export structured parser outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples. Primarily used with --stage2-parser-only.",
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
        help="Inject the Stage 5 single-domain orientation head prediction from an existing Stage 4 cache into the full SpatialBench pipeline.",
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
    return parser.parse_args()


def apply_speed_profile(profile):
    settings = {}
    if profile == "conservative":
        defaults = {
            "SOFAR_SAVE_DEBUG_ARTIFACTS": "0",
            "SOFAR_SAM_PREFER_SINGLE_MASK": "1",
            "SOFAR_FLORENCE_MAX_NEW_TOKENS": "256",
            "SOFAR_FLORENCE_NUM_BEAMS": "1",
            "SOFAR_POINTSO_VOTE_NUM": "6",
            "SOFAR_POINTSO_SAMPLE_POINTS": "4096",
            "SOFAR_QWEN_VQA_PARSE_MAX_NEW_TOKENS": "96",
            "SOFAR_QWEN_VQA_EVAL_MAX_NEW_TOKENS": "4",
        }
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
            settings[key] = os.environ[key]
    return settings


def resize_for_vlm(image, max_side):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def run_vlm_with_retry(fn, command, image, *extra_args):
    primary_max_side = int(os.getenv("SOFAR_VLM_MAX_SIDE", "896"))
    fallback_max_side = int(os.getenv("SOFAR_VLM_FALLBACK_MAX_SIDE", "640"))
    attempts = [primary_max_side]
    if fallback_max_side not in attempts:
        attempts.append(fallback_max_side)

    last_error = None
    for max_side in attempts:
        try:
            resized_image = resize_for_vlm(image, max_side=max_side)
            return fn(command, resized_image, *extra_args)
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[spatialbench] CUDA OOM at max_side={max_side}, retrying with smaller image...")
    raise last_error


def load_qwen_model():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    qwen_path = str(runtime_paths.qwen_checkpoint_path())
    qwen_dtype = resolve_qwen_dtype(torch)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=qwen_dtype
    )
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path,
        quantization_config=nf4_config,
        torch_dtype=qwen_dtype,
        attn_implementation="sdpa",
        device_map="auto",
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(qwen_path)
    return qwen_model, processor


def progress_file_path():
    return Path(output_folder) / PROGRESS_FILE


def save_progress(result):
    progress_path = progress_file_path()
    tmp_path = progress_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    tmp_path.replace(progress_path)


def load_progress(reset_progress=False):
    if reset_progress:
        return None, set()
    progress_path = progress_file_path()
    if not progress_path.exists():
        return None, set()
    with progress_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    processed_ids = {sample["id"] for sample in result.get("samples", [])}
    return result, processed_ids


def update_result(result, flag, task_type, question_type, sample_id, error=None):
    result["total"].append(flag)
    result["samples"].append({
        "id": sample_id,
        "task_type": task_type,
        "question_type": question_type,
        "correct": flag,
        "error": error,
    })
    if task_type == "position":
        if question_type == "absolute":
            result["position"]["absolute"].append(flag)
        else:
            result["position"]["relative"].append(flag)
    else:
        if question_type == "absolute":
            result["orientation"]["absolute"].append(flag)
        else:
            result["orientation"]["relative"].append(flag)


def safe_ratio(values):
    return sum(values) / max(1, len(values))


def build_summary(result, total):
    return {
        "position_relative_accuracy": safe_ratio(result["position"]["relative"]),
        "position_absolute_accuracy": safe_ratio(result["position"]["absolute"]),
        "orientation_relative_accuracy": safe_ratio(result["orientation"]["relative"]),
        "orientation_absolute_accuracy": safe_ratio(result["orientation"]["absolute"]),
        "total_accuracy": safe_ratio(result["total"]),
        "failed_samples": sum(1 for sample in result["samples"] if sample.get("error")),
        "processed_samples": len(result["samples"]),
        "remaining_samples": max(0, total - len(result["samples"])),
    }


def empty_result():
    return {
        "position": {
            "absolute": [],
            "relative": []
        },
        "orientation": {
            "absolute": [],
            "relative": []
        },
        "total": [],
        "samples": []
    }


def write_csv_records(csv_path, records, fieldnames):
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fieldnames})


def run_stage2_parser_only(args, run_id, stage2_parse_fn):
    info_list = json.load(open(spatialbench_dir / "spatial_data.json", encoding="utf-8"))
    if args.limit is not None:
        info_list = info_list[: args.limit]
    records = []
    for sample in tqdm(info_list, total=len(info_list)):
        sample_id = sample["id"]
        question = sample["question"]
        image_path = spatialbench_dir / "images" / f"{sample_id}.png"
        image = Image.open(image_path).convert("RGB")
        error = None
        parser_output = None
        try:
            parser_output = run_vlm_with_retry(stage2_parse_fn, question, image)
        except Exception as exc:
            error = str(exc)
            print(f"[stage2-parser][spatialbench] sample {sample_id} failed: {exc}")
        finally:
            del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        records.append(
            {
                "id": sample_id,
                "task_type": sample.get("task_type", ""),
                "question_type": sample.get("question_type", ""),
                "question": question,
                "target_object": (parser_output or {}).get("target_object", ""),
                "functional_part": (parser_output or {}).get("functional_part", ""),
                "relation": (parser_output or {}).get("relation", ""),
                "reference_frame": (parser_output or {}).get("reference_frame", ""),
                "reference_object": (parser_output or {}).get("reference_object", ""),
                "direction_attributes": json.dumps((parser_output or {}).get("direction_attributes", []), ensure_ascii=False),
                "parser_confidence": (parser_output or {}).get("parser_confidence"),
                "error": error,
                "parser_output": parser_output,
            }
        )

    summary = {
        "mode": "stage2_parser_only",
        "dataset": "spatialbench",
        "total_samples": len(records),
        "success_count": sum(1 for item in records if not item["error"]),
        "error_count": sum(1 for item in records if item["error"]),
        "records": records,
    }
    stable_name = "stage2_spatialbench_parser_records.json"
    stable_path, _ = write_json_outputs(summary, output_folder, stable_name, run_id)
    csv_path = stable_path.with_suffix(".csv")
    write_csv_records(
        csv_path,
        records,
        [
            "id",
            "task_type",
            "question_type",
            "question",
            "target_object",
            "functional_part",
            "relation",
            "reference_frame",
            "reference_object",
            "direction_attributes",
            "parser_confidence",
            "error",
        ],
    )
    print(f"[batch-log] wrote {csv_path}")
    return summary


def _stage3_part_query(parser_output):
    functional_part = str(parser_output.get("functional_part", "")).strip()
    if functional_part:
        return functional_part
    direction_attributes = parser_output.get("direction_attributes", [])
    if isinstance(direction_attributes, list) and direction_attributes:
        return str(direction_attributes[0]).strip()
    return ""


def run_stage3_grounding_only(args, run_id, stage2_parse_fn):
    info_list = json.load(open(spatialbench_dir / "spatial_data.json", encoding="utf-8"))
    if args.limit is not None:
        info_list = info_list[: args.limit]

    cache_root = Path(output_folder) / "stage3_spatialbench_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    records = []

    for sample in tqdm(info_list, total=len(info_list)):
        start_time = time.perf_counter()
        sample_id = sample["id"]
        question = sample["question"]
        image_path = spatialbench_dir / "images" / f"{sample_id}.png"
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        object_mask = None
        part_mask = None
        cache_paths = {}
        failed_stage = None
        status = "success"
        parser_output = None
        object_bbox = None
        part_bbox = None
        object_score = None
        part_score = None
        error = None
        part_query = ""
        timings = {
            "parser_sec": 0.0,
            "object_grounding_sec": 0.0,
            "object_sam_sec": 0.0,
            "part_grounding_sec": 0.0,
            "part_sam_sec": 0.0,
        }

        try:
            stage_start = time.perf_counter()
            parser_output = run_vlm_with_retry(stage2_parse_fn, question, image)
            timings["parser_sec"] = round(time.perf_counter() - stage_start, 2)
            target_object = str(parser_output.get("target_object", "")).strip()
            if not target_object:
                failed_stage = "parser"
                raise ValueError("Stage 3 requires a non-empty target_object from the Stage 2 parser")
            part_query = _stage3_part_query(parser_output)

            stage_start = time.perf_counter()
            object_detections = detection.get_detections(
                image,
                [target_object],
                detection_model,
                output_folder=output_folder,
                single=True,
                save_artifacts=False,
            )
            object_bbox = first_detection_xyxy(object_detections, image.width, image.height)
            object_score = first_detection_score(object_detections)
            timings["object_grounding_sec"] = round(time.perf_counter() - stage_start, 2)
            if object_bbox is None:
                failed_stage = "object_grounding"
                raise ValueError("Object grounding returned no usable bbox")

            stage_start = time.perf_counter()
            object_masks, _, _ = sam.get_mask(
                image,
                [target_object],
                sam_model,
                object_detections,
                output_folder=output_folder,
                save_artifacts=False,
                prefer_single_mask=True,
            )
            timings["object_sam_sec"] = round(time.perf_counter() - stage_start, 2)
            if len(object_masks) == 0:
                failed_stage = "object_sam"
                raise ValueError("Object SAM returned no mask")
            object_mask = object_masks[0]

            if part_query:
                roi_image_np = crop_image_array(image_np, object_bbox)
                roi_image = Image.fromarray(roi_image_np)
                stage_start = time.perf_counter()
                part_detections = detection.get_detections(
                    roi_image,
                    [part_query],
                    detection_model,
                    output_folder=output_folder,
                    single=True,
                    save_artifacts=False,
                )
                roi_part_bbox = first_detection_xyxy(part_detections, roi_image.width, roi_image.height)
                part_score = first_detection_score(part_detections)
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
                        output_folder=output_folder,
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
            cache_payload = {
                "dataset": "spatialbench",
                "sample_id": sample_id,
                "question": question,
                "task_type": sample.get("task_type", ""),
                "question_type": sample.get("question_type", ""),
                "parser_output": parser_output,
                "grounding": {
                    "target_object": parser_output.get("target_object", ""),
                    "functional_part": parser_output.get("functional_part", ""),
                    "reference_object": parser_output.get("reference_object", ""),
                    "object_bbox_xyxy": serialize_xyxy(object_bbox),
                    "object_score": object_score,
                    "part_query": part_query,
                    "part_bbox_xyxy": serialize_xyxy(part_bbox),
                    "part_score": part_score,
                    "image_size": [image.width, image.height],
                    "status": status,
                    "failed_stage": failed_stage,
                    "timings": timings,
                },
            }
            cache_paths = save_stage3_cache(cache_root / str(sample_id), cache_payload, object_mask, part_mask)
        except Exception as exc:
            error = str(exc)
            status = "error"
            print(f"[stage3-grounding][spatialbench] sample {sample_id} failed: {exc}")
            cache_payload = {
                "dataset": "spatialbench",
                "sample_id": sample_id,
                "question": question,
                "parser_output": parser_output,
                "grounding": {
                    "target_object": (parser_output or {}).get("target_object", ""),
                    "functional_part": (parser_output or {}).get("functional_part", ""),
                    "reference_object": (parser_output or {}).get("reference_object", ""),
                    "object_bbox_xyxy": serialize_xyxy(object_bbox),
                    "object_score": object_score,
                    "part_query": part_query,
                    "part_bbox_xyxy": serialize_xyxy(part_bbox),
                    "part_score": part_score,
                    "image_size": [image.width, image.height],
                    "status": status,
                    "failed_stage": failed_stage,
                    "timings": timings,
                    "error": error,
                },
            }
            cache_paths = save_stage3_cache(cache_root / str(sample_id), cache_payload, object_mask, part_mask)
        finally:
            del image, image_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_sec = round(time.perf_counter() - start_time, 2)
        records.append(
            {
                "id": sample_id,
                "task_type": sample.get("task_type", ""),
                "question_type": sample.get("question_type", ""),
                "target_object": (parser_output or {}).get("target_object", ""),
                "functional_part": (parser_output or {}).get("functional_part", ""),
                "reference_object": (parser_output or {}).get("reference_object", ""),
                "part_query": part_query,
                "status": status,
                "failed_stage": failed_stage,
                "object_bbox_xyxy": json.dumps(serialize_xyxy(object_bbox), ensure_ascii=False),
                "part_bbox_xyxy": json.dumps(serialize_xyxy(part_bbox), ensure_ascii=False),
                "object_score": object_score,
                "part_score": part_score,
                "object_mask_path": cache_paths.get("object_mask_path"),
                "part_mask_path": cache_paths.get("part_mask_path"),
                "roi_meta_path": cache_paths.get("roi_meta_path"),
                "cache_path": cache_paths.get("cache_path"),
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
        "dataset": "spatialbench",
        "total_samples": len(records),
        "success_count": sum(1 for item in records if item["status"] == "success"),
        "partial_count": sum(1 for item in records if item["status"] == "partial"),
        "error_count": sum(1 for item in records if item["status"] == "error"),
        "records": records,
    }
    stable_path, _ = write_json_outputs(summary, output_folder, "stage3_spatialbench_grounding_records.json", run_id)
    csv_path = stable_path.with_suffix(".csv")
    write_csv_records(
        csv_path,
        records,
        [
            "id",
            "task_type",
            "question_type",
            "target_object",
            "functional_part",
            "reference_object",
            "part_query",
            "status",
            "failed_stage",
            "object_bbox_xyxy",
            "part_bbox_xyxy",
            "object_score",
            "part_score",
            "object_mask_path",
            "part_mask_path",
            "roi_meta_path",
            "cache_path",
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


def run_stage4_pointdata_only(args, run_id):
    info_list = json.load(open(spatialbench_dir / "spatial_data.json", encoding="utf-8"))
    if args.limit is not None:
        info_list = info_list[: args.limit]

    depth_model = depth_esti_model.get_model()
    cache_root = Path(output_folder) / "stage3_spatialbench_cache"
    point_root = Path(output_folder) / "stage4_spatialbench_point_cache"
    point_root.mkdir(parents=True, exist_ok=True)
    sample_points_num = int(os.getenv("SOFAR_STAGE4_SAMPLE_POINTS", "4096"))
    records = []

    for sample in tqdm(info_list, total=len(info_list)):
        start_time = time.perf_counter()
        sample_id = sample["id"]
        image_path = spatialbench_dir / "images" / f"{sample_id}.png"
        cache_dir = cache_root / str(sample_id)
        cache_json = cache_dir / "object_part_cache.json"
        object_mask_path = cache_dir / "object_mask.npz"
        part_mask_path = cache_dir / "part_mask.npz"
        status = "success"
        error = None
        object_points = np.zeros((0, 6), dtype=np.float32)
        part_points = np.zeros((0, 6), dtype=np.float32)
        priors = {}
        cache_paths = {}

        try:
            if not cache_json.exists() or not object_mask_path.exists():
                raise FileNotFoundError("Stage 4 requires existing Stage 3 cache files")

            with cache_json.open("r", encoding="utf-8") as f:
                stage3_cache = json.load(f)
            object_mask = np.load(object_mask_path)["mask"]
            part_mask = np.load(part_mask_path)["mask"] if part_mask_path.exists() else None
            part_mask = constrain_child_mask_to_parent(part_mask, object_mask)

            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            depth, _, pcd = depth_esti_model.depth_estimation(image, depth_model, output_folder=output_folder)

            object_points = build_colored_points_from_mask(pcd, image_np, object_mask)
            part_points = (
                build_colored_points_from_mask(pcd, image_np, part_mask)
                if part_mask is not None and part_mask.any()
                else np.zeros((0, 6), dtype=np.float32)
            )
            object_points = sample_points(object_points, sample_points_num)
            part_points = sample_points(part_points, sample_points_num)
            priors = compute_geometry_priors(object_points, part_points)
            stage4_payload = {
                "dataset": "spatialbench",
                "sample_id": sample_id,
                "stage3_cache_path": str(cache_json),
                "parser_output": stage3_cache.get("parser_output", {}),
                "grounding": stage3_cache.get("grounding", {}),
                "geometry_priors": priors,
                "sample_points": sample_points_num,
            }
            cache_paths = save_stage4_cache(point_root / str(sample_id), stage4_payload, object_points, part_points)
        except Exception as exc:
            error = str(exc)
            status = "error"
            print(f"[stage4-pointdata][spatialbench] sample {sample_id} failed: {exc}")

        total_sec = round(time.perf_counter() - start_time, 2)
        records.append(
            {
                "id": sample_id,
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
        "dataset": "spatialbench",
        "total_samples": len(records),
        "success_count": sum(1 for item in records if item["status"] == "success"),
        "error_count": sum(1 for item in records if item["status"] == "error"),
        "records": records,
    }
    stable_path, _ = write_json_outputs(summary, output_folder, "stage4_spatialbench_point_records.json", run_id)
    csv_path = stable_path.with_suffix(".csv")
    write_csv_records(
        csv_path,
        records,
        [
            "id",
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


def recover_progress_from_log(log_path, info_list):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Recover log not found: {log_path}")

    recovered = empty_result()
    sample_index = -1
    waiting_answer = False
    letters = "ABCD"

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if "info: [" in line:
                sample_index += 1
                waiting_answer = sample_index < len(info_list)
                continue
            if not waiting_answer:
                continue
            if line in {"A", "B", "C", "D"}:
                sample = info_list[sample_index]
                gold = letters[sample["answer"]]
                flag = line == gold
                update_result(
                    recovered,
                    flag,
                    sample["task_type"],
                    sample["question_type"],
                    sample["id"],
                    error=None,
                )
                waiting_answer = False

    processed_ids = {sample["id"] for sample in recovered["samples"]}
    return recovered, processed_ids


def process_info(info):
    id = info["id"]

    question = info["question"]
    options = info["options"]
    answer = info["answer"]
    task_type = info["task_type"]
    question_type = info["question_type"]

    prompt = question + "\n" + "A. " + options[0] + "\n" + "B. " + options[1] + "\n" + "C. " + options[
        2] + "\n" + "D. " + options[3]

    image_path = spatialbench_dir / "images" / f"{id}.png"
    image = Image.open(image_path).convert("RGB")
    detections = None
    mask = None
    ann_img = None
    object_names = None
    depth = None
    pcd = None
    scene_graph = None
    text = None
    stage5_prediction = None
    stage5_context = None
    stage5_gate = None

    try:
        info = run_vlm_with_retry(parse_fn, prompt, image)
        print(json.dumps(info, indent=2))
        object_list = list(info.keys())

        detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
        mask, ann_img, object_names = sam.get_mask(
            image, object_list, sam_model, detections, output_folder=output_folder)

        depth, _, pcd = depth_esti_model.depth_estimation(image, depth_model, output_folder=output_folder)
        scene_graph, _ = get_scene_graph(
            image, pcd, mask, info, object_names, orientation_model, output_folder=output_folder
        )
        if STAGE5_OPTIONS.get("enabled"):
            stage5_gate = classify_spatialbench_stage5_applicability(question, parser_info=info)
            if stage5_gate.get("applicable"):
                stage5_prediction = predict_from_stage4_dir(
                    Path(output_folder) / "stage4_spatialbench_point_cache" / str(id),
                    dataset="spatialbench",
                    checkpoint_path=STAGE5_OPTIONS.get("checkpoint_path"),
                    device=STAGE5_OPTIONS.get("device", "auto"),
                    num_points=STAGE5_OPTIONS.get("num_points", 1024),
                )
                if stage5_prediction:
                    scene_graph = inject_prediction_into_scene_graph(scene_graph, stage5_prediction)
                    stage5_context = build_spatialbench_stage5_context(question, stage5_gate, stage5_prediction)
                    print(
                        f"[spatialbench] sample {id} stage5 apply "
                        f"category={stage5_gate.get('category')} "
                        f"direction={stage5_prediction.get('direction_vector')}"
                    )
                else:
                    print(
                        f"[spatialbench] sample {id} stage5 skipped "
                        f"category={stage5_gate.get('category')} "
                        "reason=prediction unavailable"
                    )
            else:
                print(
                    f"[spatialbench] sample {id} stage5 skipped "
                    f"category={stage5_gate.get('category')} "
                    f"reason={stage5_gate.get('reason')}"
                )
        text = run_vlm_with_retry(reason_fn, prompt, ann_img, scene_graph, True, stage5_context)

        print(text)
        if ("A" == text[0] and answer == 0) or ("B" == text[0] and answer == 1) or ("C" == text[0] and answer == 2) or (
                "D" == text[0] and answer == 3):
            return True, task_type, question_type, id
        else:
            return False, task_type, question_type, id
    finally:
        del image, detections, mask, ann_img, object_names, depth, pcd, scene_graph, text
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    selected_modes = sum(
        1
        for flag in (
            args.stage2_parser_only,
            args.stage3_grounding_only,
            args.stage4_pointdata_only,
        )
        if flag
    )
    if selected_modes > 1:
        raise ValueError("Only one of --stage2-parser-only / --stage3-grounding-only / --stage4-pointdata-only can be used at a time.")
    speed_profile_settings = apply_speed_profile(args.speed_profile)
    if args.stage2_parser_only:
        log_prefix = "stage2_spatialbench_parser"
    elif args.stage3_grounding_only:
        log_prefix = "stage3_spatialbench_grounding"
    elif args.stage4_pointdata_only:
        log_prefix = "stage4_spatialbench_pointdata"
    else:
        log_prefix = "eval_spatialbench"
    run_id, _ = setup_timestamped_logging(output_folder, log_prefix)
    if args.speed_profile != "off":
        print(f"[spatialbench] speed_profile={args.speed_profile}")
        print(f"[spatialbench] speed_profile_settings={speed_profile_settings}")
    llm_backend = resolve_llm_backend()
    print(f"LLM backend: {llm_backend}")

    if llm_backend == "qwen":
        from serve.qwen_inference import stage2_part_parser as qwen_stage2_part_parser
        from serve.qwen_inference import vqa_parsing as qwen_vqa_parsing
        from serve.qwen_inference import vqa_spatial_reasoning as qwen_vqa_spatial_reasoning
        qwen_model, processor = load_qwen_model()

        def parse_fn(command, image):
            return qwen_vqa_parsing(qwen_model, processor, image, command)

        def reason_fn(command, image, scene_graph, eval, stage5_context=None):
            return qwen_vqa_spatial_reasoning(
                qwen_model,
                processor,
                image,
                command,
                scene_graph,
                eval=eval,
                stage5_context=stage5_context,
            )

        def stage2_parse_fn(command, image):
            return qwen_stage2_part_parser(qwen_model, processor, image, command)
    else:
        from serve.chatgpt import vqa_parsing as chatgpt_vqa_parsing
        from serve.chatgpt import vqa_spatial_reasoning as chatgpt_vqa_spatial_reasoning

        def parse_fn(command, image):
            return chatgpt_vqa_parsing(command, image)

        def reason_fn(command, image, scene_graph, eval, stage5_context=None):
            return chatgpt_vqa_spatial_reasoning(image, command, scene_graph, eval=eval)

        def stage2_parse_fn(command, image):
            raise NotImplementedError("Stage 2 parser smoke is currently implemented for the qwen backend only.")

    if args.stage2_parser_only:
        run_stage2_parser_only(args, run_id, stage2_parse_fn)
        raise SystemExit(0)
    if args.stage3_grounding_only:
        run_stage3_grounding_only(args, run_id, stage2_parse_fn)
        raise SystemExit(0)
    if args.stage4_pointdata_only:
        run_stage4_pointdata_only(args, run_id)
        raise SystemExit(0)

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    depth_model = depth_esti_model.get_model()
    orientation_model = orientation.get_model()
    STAGE5_OPTIONS.clear()
    STAGE5_OPTIONS.update(
        {
            "enabled": bool(args.use_stage5_head),
            "checkpoint_path": args.stage5_checkpoint,
            "device": args.stage5_device,
            "num_points": args.stage5_num_points,
        }
    )
    if STAGE5_OPTIONS["enabled"]:
        print("[spatialbench] stage5_head=enabled")
        if STAGE5_OPTIONS["checkpoint_path"]:
            print(f"[spatialbench] stage5_checkpoint={STAGE5_OPTIONS['checkpoint_path']}")
        print(f"[spatialbench] stage5_device={STAGE5_OPTIONS['device']}")
        print(f"[spatialbench] stage5_num_points={STAGE5_OPTIONS['num_points']}")

    info_list = json.load(open(spatialbench_dir / "spatial_data.json"))
    total = len(info_list)
    print("total: ", total)
    result, processed_ids = load_progress(reset_progress=args.reset_progress)
    if args.reset_progress:
        print("[spatialbench] starting fresh because --reset-progress was provided")
    if result is None:
        if args.recover_log:
            result, processed_ids = recover_progress_from_log(args.recover_log, info_list)
            print(f"[spatialbench] recovered {len(processed_ids)} completed samples from log {args.recover_log}")
            save_progress(result)
        else:
            result = empty_result()
            processed_ids = set()
    else:
        print(f"[spatialbench] resuming from progress file with {len(processed_ids)} completed samples")

    pending_info_list = [sample for sample in info_list if sample["id"] not in processed_ids]
    if args.limit is not None:
        pending_info_list = pending_info_list[: args.limit]
        print(f"[spatialbench] applying --limit {args.limit}, runnable samples reduced to {len(pending_info_list)}")
    print(f"[spatialbench] pending samples: {len(pending_info_list)}")

    if llm_backend == "qwen":
        iterator = tqdm(pending_info_list, total=len(pending_info_list))
        for sample in iterator:
            try:
                flag, task_type, question_type, sample_id = process_info(sample)
                error = None
            except Exception as exc:
                sample_id = sample.get("id")
                task_type = sample.get("task_type")
                question_type = sample.get("question_type")
                flag = False
                error = str(exc)
                print(f"[spatialbench] sample {sample_id} failed: {exc}")
            update_result(result, flag, task_type, question_type, sample_id, error=error)
            save_progress(result)
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_info, info): info for info in pending_info_list}
            for future in tqdm(as_completed(futures), total=len(pending_info_list)):
                sample = futures[future]
                try:
                    flag, task_type, question_type, sample_id = future.result()
                    error = None
                except Exception as exc:
                    sample_id = sample.get("id")
                    task_type = sample.get("task_type")
                    question_type = sample.get("question_type")
                    flag = False
                    error = str(exc)
                    print(f"[spatialbench] sample {sample_id} failed: {exc}")
                update_result(result, flag, task_type, question_type, sample_id, error=error)
                save_progress(result)

    summary = build_summary(result, total)
    print("Position relative accuracy: ", summary["position_relative_accuracy"])
    print("Position absolute accuracy: ", summary["position_absolute_accuracy"])
    print("Orientation relative accuracy: ", summary["orientation_relative_accuracy"])
    print("Orientation absolute accuracy: ", summary["orientation_absolute_accuracy"])
    print("Total accuracy: ", summary["total_accuracy"])

    result["summary"] = summary
    save_progress(result)
    write_json_outputs(result, output_folder, "eval_spatialbench.json", run_id)
