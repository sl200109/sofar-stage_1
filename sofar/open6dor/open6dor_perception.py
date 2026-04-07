import os
import glob
import json
import warnings
import numpy as np
import sys
import time
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve import pointso as orientation
from utils import preprocess_open6dor_image
from serve.scene_graph import open6dor_scene_graph
from segmentation import sam, florence as detection
from serve.utils import generate_rotation_matrix, get_point_cloud_from_rgbd
from serve import runtime_paths
from serve.batch_logging import setup_timestamped_logging, write_json_outputs
from serve.qwen_runtime import resolve_qwen_dtype

warnings.filterwarnings("ignore")
open6dor_parsing_fn = None
open6dor_spatial_reasoning_fn = None
detection_model = None
sam_model = None
orientation_model = None
RUN_RECORDS = []
PROGRESS_FILE = "open6dor_perception_progress.json"


def resolve_llm_backend():
    backend = os.getenv("SOFAR_LLM_BACKEND", "qwen").strip().lower()
    if backend not in {"openai", "qwen"}:
        raise ValueError(f"Unsupported SOFAR_LLM_BACKEND: {backend}")
    return backend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N pending task directories after resume/skip filtering.",
    )
    return parser.parse_args()


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


def progress_file_path(output_dir):
    return Path(output_dir) / PROGRESS_FILE


def save_progress(output_dir, run_id, dataset_root, dataset_paths):
    summary = {
        "run_id": run_id,
        "dataset_root": str(dataset_root),
        "total_tasks": len(dataset_paths),
        "success_count": sum(1 for item in RUN_RECORDS if item["status"] == "success"),
        "error_count": sum(1 for item in RUN_RECORDS if item["status"] == "error"),
        "skipped_count": sum(1 for item in RUN_RECORDS if item["status"] == "skipped"),
        "processed_count": len(RUN_RECORDS),
        "remaining_count": max(0, len(dataset_paths) - len(RUN_RECORDS)),
        "records": RUN_RECORDS,
    }
    progress_path = progress_file_path(output_dir)
    tmp_path = progress_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    tmp_path.replace(progress_path)
    return summary


def load_progress(output_dir):
    progress_path = progress_file_path(output_dir)
    if not progress_path.exists():
        return [], set()
    with progress_path.open("r", encoding="utf-8") as f:
        progress = json.load(f)
    records = progress.get("records", [])
    processed = {item["task_dir"] for item in records}
    return records, processed


def should_skip_existing_result(task_dir):
    if os.getenv("SOFAR_OPEN6DOR_FORCE_RERUN", "").strip() == "1":
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


def process_dataset(p):
    start_time = time.perf_counter()
    try:
        image_path = os.path.join(p, "isaac_render-rgb-0-1.png")
        depth_path = os.path.join(p, "isaac_render-depth-0-1.npy")
        info = json.load(open(os.path.join(p, "task_config_new5.json")))
        prompt = f"Pick the {info['target_obj_name']}. " + info["instruction"] if "rot" in p else info["instruction"]
        print(prompt)

        output_path = os.path.join(p, "output")
        os.makedirs(output_path, exist_ok=True)

        image = Image.open(image_path).convert("RGB")

        depth = np.load(depth_path)
        vinvs = np.array([[0., 1., 0., 0.],
                          [-0.9028605, -0., 0.42993355, -0.],
                          [0.42993355, -0., 0.9028605, -0.],
                          [1., 0., 1.2, 1.]])
        projs = [[1.7320507, 0., 0., 0.],
                 [0., 2.5980759, 0., 0.],
                 [0., 0., 0., -1.],
                 [0., 0., 0.05, 0.]]
        pcd = get_point_cloud_from_rgbd(depth, np.array(image), vinvs, projs).cpu().numpy().astype(np.float64)
        pcd = pcd.reshape(depth.shape[0], depth.shape[1], 6)[:, :, :3]

        info = open6dor_parsing_fn(prompt, image)
        info['related_objects'] = [] if "rot" in p else info['related_objects']
        object_list = [info['picked_object']] + info['related_objects']
        print(info)

        image = preprocess_open6dor_image(image)
        detections = detection.get_detections(image, object_list, detection_model, output_folder=output_path, single=True)
        mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_path)

        picked_object_info, other_objects_info, picked_object_dict \
            = open6dor_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_path)

        if "rot" not in p:
            response = open6dor_spatial_reasoning_fn(image, prompt, picked_object_info, other_objects_info)
            init_position = picked_object_dict["center"]
            target_position = response["target_position"]
            print(response)
        else:
            init_position = picked_object_dict["center"]
            target_position = picked_object_dict["center"]

        init_orientation = picked_object_dict["orientation"]
        target_orientation = info["target_orientation"]

        if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
            direction_attributes = target_orientation.keys()
            init_directions = [init_orientation[direction] for direction in direction_attributes]
            target_directions = [target_orientation[direction] for direction in direction_attributes]
            transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
        else:
            transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        result = {
            'task_dir': p,
            'init_position': init_position,
            'target_position': target_position,
            'delta_position': [round(target_position[i] - init_position[i], 2) for i in range(3)],
            'init_orientation': init_orientation,
            'target_orientation': target_orientation,
            'transform_matrix': transform_matrix
        }
        with open(os.path.join(output_path, "result.json"), 'w') as f:
            json.dump(result, f, indent=4)
        print("Successfully saved result for", output_path)
        return {
            "task_dir": p,
            "status": "success",
            "result_file": os.path.join(output_path, "result.json"),
            "elapsed_sec": round(time.perf_counter() - start_time, 2),
        }

    except Exception as e:
        print(f"Error processing {p}: {e}")
        return {
            "task_dir": p,
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(time.perf_counter() - start_time, 2),
        }


if __name__ == "__main__":
    args = parse_args()
    output_dir = runtime_paths.ensure_output_dir()
    run_id, _ = setup_timestamped_logging(output_dir, "open6dor_perception")
    dataset_root = runtime_paths.open6dor_dataset_dir()
    dataset_paths = discover_task_dirs(dataset_root)
    RUN_RECORDS, processed_task_dirs = load_progress(output_dir)
    if processed_task_dirs:
        print(f"[open6dor] resuming from progress file with {len(processed_task_dirs)} completed tasks")
    print(f"[open6dor] discovered {len(dataset_paths)} task directories")

    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()
    llm_backend = resolve_llm_backend()
    print(f"LLM backend: {llm_backend}")

    if llm_backend == "qwen":
        from serve.qwen_inference import open6dor_parsing as qwen_open6dor_parsing
        from serve.qwen_inference import open6dor_spatial_reasoning as qwen_open6dor_spatial_reasoning
        qwen_model, processor = load_qwen_model()

        def open6dor_parsing_fn(command, img):
            return qwen_open6dor_parsing(qwen_model, processor, img, command)

        def open6dor_spatial_reasoning_fn(img, command, picked_info, other_info):
            return qwen_open6dor_spatial_reasoning(qwen_model, processor, img, command, picked_info, other_info)
    else:
        from serve.chatgpt import open6dor_parsing as chatgpt_open6dor_parsing
        from serve.chatgpt import open6dor_spatial_reasoning as chatgpt_open6dor_spatial_reasoning

        def open6dor_parsing_fn(command, img):
            return chatgpt_open6dor_parsing(command, img)

        def open6dor_spatial_reasoning_fn(img, command, picked_info, other_info):
            return chatgpt_open6dor_spatial_reasoning(img, command, picked_info, other_info)

    # for p in tqdm(dataset_paths):
    #     process_dataset(p)
    pending_dataset_paths = [p for p in dataset_paths if p not in processed_task_dirs]
    skipped_existing = []
    runnable_dataset_paths = []
    for p in pending_dataset_paths:
        if should_skip_existing_result(p):
            skipped_existing.append({
                "task_dir": p,
                "status": "skipped",
                "result_file": os.path.join(p, "output", "result.json"),
                "elapsed_sec": 0.0,
            })
        else:
            runnable_dataset_paths.append(p)
    if skipped_existing:
        RUN_RECORDS.extend(skipped_existing)
        save_progress(output_dir, run_id, dataset_root, dataset_paths)
        print(f"[open6dor] skipped {len(skipped_existing)} tasks with existing result.json")
    if args.limit is not None:
        runnable_dataset_paths = runnable_dataset_paths[:args.limit]
        print(f"[open6dor] applying --limit {args.limit}, runnable tasks reduced to {len(runnable_dataset_paths)}")
    print(f"[open6dor] pending runnable tasks: {len(runnable_dataset_paths)}")

    if llm_backend == "qwen":
        for p in tqdm(runnable_dataset_paths):
            record = process_dataset(p)
            RUN_RECORDS.append(record)
            save_progress(output_dir, run_id, dataset_root, dataset_paths)
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for record in tqdm(executor.map(process_dataset, runnable_dataset_paths), total=len(runnable_dataset_paths)):
                RUN_RECORDS.append(record)
                save_progress(output_dir, run_id, dataset_root, dataset_paths)

    summary = save_progress(output_dir, run_id, dataset_root, dataset_paths)
    write_json_outputs(summary, output_dir, "open6dor_perception_summary.json", run_id)
