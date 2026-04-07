import os
import json
import warnings
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve import pointso as orientation
from serve.scene_graph import open6dor_scene_graph
from segmentation import sam, florence as detection
from serve.utils import generate_rotation_matrix, get_point_cloud_from_rgbd
from serve import runtime_paths

warnings.filterwarnings("ignore")
output_dir = runtime_paths.ensure_output_dir()
output_folder = str(output_dir)


def resolve_llm_backend():
    backend = os.getenv("SOFAR_LLM_BACKEND", "qwen").strip().lower()
    if backend not in {"openai", "qwen"}:
        raise ValueError(f"Unsupported SOFAR_LLM_BACKEND: {backend}")
    return backend


def load_qwen_model():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    qwen_path = str(runtime_paths.qwen_checkpoint_path())
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path,
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(qwen_path)
    return qwen_model, processor

if __name__ == "__main__":
    image_path = "assets/open6dor.png"
    depth_path = "assets/open6dor.npy"
    prompt = "Place the knife behind the clipboard on the table. And rotate the handle of the knife to left."

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
    scene_path = output_dir / "scene.npy"
    np.save(scene_path, pcd)
    pcd = pcd.reshape(depth.shape[0], depth.shape[1], 6)[:, :, :3]

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()
    llm_backend = resolve_llm_backend()
    print(f"LLM backend: {llm_backend}")

    if llm_backend == "qwen":
        from serve.qwen_inference import open6dor_parsing as qwen_open6dor_parsing
        from serve.qwen_inference import open6dor_spatial_reasoning as qwen_open6dor_spatial_reasoning
        qwen_model, processor = load_qwen_model()

        def parse_fn(cmd, img):
            return qwen_open6dor_parsing(qwen_model, processor, img, cmd)

        def reason_fn(img, cmd, picked_info, other_info):
            return qwen_open6dor_spatial_reasoning(qwen_model, processor, img, cmd, picked_info, other_info)
    else:
        from serve.chatgpt import open6dor_parsing as chatgpt_open6dor_parsing
        from serve.chatgpt import open6dor_spatial_reasoning as chatgpt_open6dor_spatial_reasoning

        def parse_fn(cmd, img):
            return chatgpt_open6dor_parsing(cmd, img)

        def reason_fn(img, cmd, picked_info, other_info):
            return chatgpt_open6dor_spatial_reasoning(img, cmd, picked_info, other_info)

    print("Start object parsing...")
    info = parse_fn(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = [info['picked_object']] + info['related_objects']

    print("Start Segment Anything...")
    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder, single=True)
    mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_folder)
    if set(object_list) != set(object_names):
        raise ValueError("Grounded SAM Error: object list does not match the detected object names.")

    print("Generate scene graph...")
    picked_object_info, other_objects_info, picked_object_dict \
        = open6dor_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_folder)
    print("picked info:", picked_object_info)
    print("other info:")
    for node in other_objects_info:
        print(node)

    print("Start spatial reasoning...")
    response = reason_fn(image, prompt, picked_object_info, other_objects_info)
    print(response)

    init_position = picked_object_dict["center"]
    target_position = response["target_position"]
    init_orientation = picked_object_dict["orientation"]
    target_orientation = info["target_orientation"]

    if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
        direction_attributes = target_orientation.keys()
        init_directions = [init_orientation[direction] for direction in direction_attributes]
        target_directions = [target_orientation[direction] for direction in direction_attributes]
        transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
    else:
        transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    print("Result:")
    result = {
        'init_position': init_position,
        'target_position': target_position,
        'delta_position': [round(target_position[i] - init_position[i], 2) for i in range(3)],
        'init_orientation': init_orientation,
        'target_orientation': target_orientation,
        'transform_matrix': transform_matrix
    }
    print(result)
    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_payload = {
        "timestamp": run_stamp,
        "llm_backend": llm_backend,
        "prompt": prompt,
        "image_path": image_path,
        "depth_path": depth_path,
        "scene_path": str(scene_path),
        "parsed_info": info,
        "picked_object_info": picked_object_info,
        "other_objects_info": other_objects_info,
        "reasoning_response": response,
        "result": result,
        "output_dir": str(output_dir),
    }
    log_path_latest = output_dir / "open6dor_demo_log.json"
    log_path_stamped = output_dir / f"open6dor_demo_log_{run_stamp}.json"
    with open(log_path_latest, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False)
    with open(log_path_stamped, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved scene: {scene_path}")
    print(f"Saved result: {result_path}")
    print(f"Saved log: {log_path_latest}")
    print(f"Saved log snapshot: {log_path_stamped}")
