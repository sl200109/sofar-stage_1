import os
import argparse
import json
import warnings
import sys
import gc
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

warnings.filterwarnings("ignore")
output_folder = str(runtime_paths.ensure_output_dir())
spatialbench_dir = runtime_paths.spatialbench_dataset_dir()
parse_fn = None
reason_fn = None
PROGRESS_FILE = "eval_spatialbench_progress.json"


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
        text = run_vlm_with_retry(reason_fn, prompt, ann_img, scene_graph, True)

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
    speed_profile_settings = apply_speed_profile(args.speed_profile)
    run_id, _ = setup_timestamped_logging(output_folder, "eval_spatialbench")
    if args.speed_profile != "off":
        print(f"[spatialbench] speed_profile={args.speed_profile}")
        print(f"[spatialbench] speed_profile_settings={speed_profile_settings}")
    llm_backend = resolve_llm_backend()
    print(f"LLM backend: {llm_backend}")

    if llm_backend == "qwen":
        from serve.qwen_inference import vqa_parsing as qwen_vqa_parsing
        from serve.qwen_inference import vqa_spatial_reasoning as qwen_vqa_spatial_reasoning
        qwen_model, processor = load_qwen_model()

        def parse_fn(command, image):
            return qwen_vqa_parsing(qwen_model, processor, image, command)

        def reason_fn(command, image, scene_graph, eval):
            return qwen_vqa_spatial_reasoning(qwen_model, processor, image, command, scene_graph, eval=eval)
    else:
        from serve.chatgpt import vqa_parsing as chatgpt_vqa_parsing
        from serve.chatgpt import vqa_spatial_reasoning as chatgpt_vqa_spatial_reasoning

        def parse_fn(command, image):
            return chatgpt_vqa_parsing(command, image)

        def reason_fn(command, image, scene_graph, eval):
            return chatgpt_vqa_spatial_reasoning(image, command, scene_graph, eval=eval)

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    depth_model = depth_esti_model.get_model()
    orientation_model = orientation.get_model()

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
