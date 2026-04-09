import os
import numpy as np
from typing import Union, Any, Tuple, Dict
from unittest.mock import patch

import torch
from PIL import Image
import supervision as sv
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# florance-2 checkpoint
FLORENCE_CHECKPOINT = "microsoft/Florence-2-base"
FLORENCE_OPEN_VOCABULARY_DETECTION_TASK = '<OPEN_VOCABULARY_DETECTION>'


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    # imports.remove("flash_attn")
    return imports


def get_model():
    local_florence_path = "checkpoints/Florence-2-base"
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            local_florence_path, trust_remote_code=True).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(
            local_florence_path, trust_remote_code=True)
        return model, processor


def run_florence_inference(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image,
    task: str,
    text: str = ""
) -> Tuple[str, Dict]:
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    max_new_tokens = _env_int("SOFAR_FLORENCE_MAX_NEW_TOKENS", 1024)
    num_beams = _env_int("SOFAR_FLORENCE_NUM_BEAMS", 3)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(
        generated_text, task=task, image_size=image.size)
    return generated_text, response


def get_detections(image, obj_list, florence_model, output_folder="output", single=False, save_artifacts=None):

    model, processor = florence_model
    if save_artifacts is None:
        save_artifacts = _env_flag("SOFAR_SAVE_DEBUG_ARTIFACTS", True)

    detections_list = []
    for i in range(len(obj_list)):
        obj = obj_list[i]
        _, result = run_florence_inference(
            model=model,
            processor=processor,
            device=DEVICE,
            image=image,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=obj
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image.size
        )
        detections.class_id = np.full(len(detections.xyxy), i)
        detections.confidence = np.full(len(detections.xyxy), 1.0)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)

    image = np.array(image)
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        thickness=1
    )
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

    # save the annotated florence image
    if save_artifacts:
        annotated_frame = Image.fromarray(annotated_frame)
        annotated_frame.save(os.path.join(output_folder, "florence_image.jpg"))

    if single:
        # Filter detections to only include highest confidence detections per object
        print(f"Before Selection: {len(detections.xyxy)} boxes")

        # Select highest confidence detection for each object in object_list
        highest_confidence_detections = {}
        for i, class_id in enumerate(detections.class_id):
            confidence = detections.confidence[i]
            if class_id not in highest_confidence_detections or confidence > highest_confidence_detections[class_id][1]:
                highest_confidence_detections[class_id] = (i, confidence)

        # Filter detections to only include highest confidence detections per object
        selected_indices = [idx for idx, _ in highest_confidence_detections.values()]
        detections.xyxy = detections.xyxy[selected_indices]
        detections.confidence = detections.confidence[selected_indices]
        detections.class_id = detections.class_id[selected_indices]

        print(f"After Selection: {len(detections.xyxy)} boxes")

    return detections
