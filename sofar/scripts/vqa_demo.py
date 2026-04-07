import os
import json
import warnings
import numpy as np
import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve import pointso as orientation
from depth import metric3dv2 as depth_model
from serve.scene_graph import get_scene_graph
from segmentation import sam, florence as detection
from serve import runtime_paths

warnings.filterwarnings("ignore")
output_folder = str(runtime_paths.ensure_output_dir())


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
    image_path = "assets/table.jpg"
    prompt = "How far between the left bottle and the right bottle?"

    image = Image.open(image_path).convert("RGB")

    print("Load models...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    orientation_model = orientation.get_model()
    metriced_model = depth_model.get_model()
    llm_backend = resolve_llm_backend()
    print(f"LLM backend: {llm_backend}")

    if llm_backend == "qwen":
        from serve.qwen_inference import vqa_parsing as qwen_vqa_parsing
        from serve.qwen_inference import vqa_spatial_reasoning as qwen_vqa_spatial_reasoning
        qwen_model, processor = load_qwen_model()

        def parse_fn(cmd, img):
            return qwen_vqa_parsing(qwen_model, processor, img, cmd)

        def reason_fn(img, cmd, scene_graph):
            return qwen_vqa_spatial_reasoning(qwen_model, processor, img, cmd, scene_graph, eval=False)
    else:
        from serve.chatgpt import vqa_parsing as chatgpt_vqa_parsing
        from serve.chatgpt import vqa_spatial_reasoning as chatgpt_vqa_spatial_reasoning

        def parse_fn(cmd, img):
            return chatgpt_vqa_parsing(cmd, img)

        def reason_fn(img, cmd, scene_graph):
            return chatgpt_vqa_spatial_reasoning(img, cmd, scene_graph)

    print("Start object parsing...")
    info = parse_fn(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    print("Start Segment Anything...")
    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_folder)

    print("Predict depth map...")
    depth, _, pcd = depth_model.depth_estimation(image, metriced_model, output_folder=output_folder)
    np.save(os.path.join(output_folder, "scene.npy"), np.concatenate([pcd, np.array(image)], axis=-1).reshape(-1, 6))

    print("Generate scene graph...")
    scene_graph, _ = get_scene_graph(image, pcd, mask, info, object_names, orientation_model, output_folder=output_folder)

    print("objects info:")
    for node in scene_graph:
        print(node)

    print("Start spatial reasoning...")
    response = reason_fn(ann_img, prompt, scene_graph)
    print(response)
