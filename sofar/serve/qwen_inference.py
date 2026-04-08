import json
import re
import os
from qwen_vl_utils import process_vision_info
from serve.system_prompts import (
    open6dor_parsing_prompt,
    open6dor_reasoning_prompt,
    manip_parsing_prompt,
    manip_reasoning_prompt,
    vqa_parsing_prompt,
    vqa_reasoning_prompt,
)


def _extract_json_block_from_index(text, start):
    opening = text[start]
    if opening not in "{[":
        return None
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _iter_json_block_candidates(text):
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        candidate = _extract_json_block_from_index(text, idx)
        if candidate:
            yield candidate


def _repair_invalid_backslash_escape(text):
    # Keep valid JSON escapes, repair model outputs like "\( ... \)".
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _parse_numeric_triplet(text):
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(matches) != 3:
        return None
    return [float(value) for value in matches]


def _load_qwen_json(output_text):
    cleaned = output_text.replace("```json", "").replace("```", "").strip()
    cleaned = _repair_invalid_backslash_escape(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for candidate in _iter_json_block_candidates(cleaned):
        try:
            return json.loads(_repair_invalid_backslash_escape(candidate))
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("No valid JSON object found in model output", cleaned, 0)


def _load_open6dor_reasoning_json(output_text):
    cleaned = output_text.replace("```json", "").replace("```", "").strip()
    cleaned = _repair_invalid_backslash_escape(cleaned)

    # Prefer dict results with target_position.
    for candidate in _iter_json_block_candidates(cleaned):
        try:
            obj = json.loads(_repair_invalid_backslash_escape(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "target_position" in obj:
            return obj

    # Fallback: list like [x, y, z] if model only outputs position array.
    for candidate in _iter_json_block_candidates(cleaned):
        try:
            obj = json.loads(_repair_invalid_backslash_escape(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and len(obj) == 3:
            return {
                "calculation_process": "",
                "target_position": obj,
            }

    # Fallback: recover a coordinate triplet from natural-language answers.
    position_patterns = [
        r"(?is)(?:target|final)\s+position[^[]*(\[[^\]]+\])",
        r"(?is)position\s+for\s+the\s+.+?\s+is[^[]*(\[[^\]]+\])",
        r"(?is)thus,\s+the\s+target\s+position[^[]*(\[[^\]]+\])",
    ]
    for pattern in position_patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        triplet = _parse_numeric_triplet(match.group(1))
        if triplet is not None:
            return {
                "calculation_process": "",
                "target_position": triplet,
            }

    bracket_triplets = re.findall(r"\[[^\[\]]+\]", cleaned)
    for candidate in reversed(bracket_triplets):
        triplet = _parse_numeric_triplet(candidate)
        if triplet is not None:
            return {
                "calculation_process": "",
                "target_position": triplet,
            }

    raise json.JSONDecodeError("No valid Open6DOR reasoning JSON found in model output", cleaned, 0)


def _load_manip_reasoning_json(output_text):
    cleaned = output_text.replace("```json", "").replace("```", "").strip()
    cleaned = _repair_invalid_backslash_escape(cleaned)

    for candidate in _iter_json_block_candidates(cleaned):
        try:
            obj = json.loads(_repair_invalid_backslash_escape(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "target_position" in obj:
            obj.setdefault("interact_object_id", 1)
            obj.setdefault("target_orientation", {})
            return obj

    raise json.JSONDecodeError("No valid manipulation reasoning JSON found in model output", cleaned, 0)


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _qwen_generate_text(qwen_model, processor, messages, max_new_tokens=1024):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def _normalize_object_direction_info(raw_info):
    if isinstance(raw_info, dict):
        if "info" in raw_info and isinstance(raw_info["info"], list):
            info = {}
            for item in raw_info["info"]:
                if not isinstance(item, dict):
                    continue
                object_name = item.get("object_name") or item.get("name")
                if not object_name:
                    continue
                direction_attributes = item.get("direction_attributes", [])
                if not isinstance(direction_attributes, list):
                    direction_attributes = []
                info[object_name] = direction_attributes
            if info:
                return info
        if "object_list" in raw_info and isinstance(raw_info["object_list"], list):
            return {name: [] for name in raw_info["object_list"] if isinstance(name, str)}
        if all(isinstance(k, str) for k in raw_info.keys()):
            normalized = {}
            for key, value in raw_info.items():
                normalized[key] = value if isinstance(value, list) else []
            if normalized:
                return normalized
    if isinstance(raw_info, list):
        if raw_info and all(isinstance(item, dict) for item in raw_info):
            info = {}
            for item in raw_info:
                object_name = item.get("object_name") or item.get("name")
                if not object_name:
                    continue
                direction_attributes = item.get("direction_attributes", [])
                if not isinstance(direction_attributes, list):
                    direction_attributes = []
                info[object_name] = direction_attributes
            if info:
                return info
        return {name: [] for name in raw_info if isinstance(name, str)}
    raise ValueError(f"Unsupported parsed object schema: {raw_info}")


def _normalize_target_orientation(target_orientation):
    if isinstance(target_orientation, dict):
        return target_orientation
    if isinstance(target_orientation, list):
        normalized = {}
        for item in target_orientation:
            if not isinstance(item, dict):
                continue
            key = item.get("direction_attributes") or item.get("direction_attribute")
            val = item.get("value") or item.get("target_direction")
            if key is not None and val is not None:
                normalized[key] = val
        return normalized
    return {}


def _extract_choice_letter(text):
    for ch in text.strip().upper():
        if ch in {"A", "B", "C", "D"}:
            return ch
    return text.strip()


def open6dor_parsing(qwen_model, processor, image_path, instruction):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": open6dor_parsing_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]

    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int("SOFAR_QWEN_OPEN6DOR_PARSE_MAX_NEW_TOKENS", 256),
    )
    print(output_text)
    info = _load_qwen_json(output_text)
    directions = {}
    direction_attributes = []
    for direction in info['direction']:
        directions[direction['direction_attribute']] = direction['target_direction']
        direction_attributes.append(direction['direction_attribute'])
    info['target_orientation'] = directions
    info['direction_attributes'] = direction_attributes
    print(info)
    return info


def open6dor_spatial_reasoning(qwen_model, processor, image_path, instruction, picked_object_info, other_objects_info):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": open6dor_reasoning_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": (
                        f"Command: {instruction}\n"
                        f"picked_object_info: {picked_object_info}\n"
                        f"other_objects_info: {other_objects_info}\n"
                        'Return JSON only in the format '
                        '{"calculation_process": "...", "target_position": [x, y, z]}.'
                    )
                },
            ],
        }
    ]

    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int("SOFAR_QWEN_OPEN6DOR_REASON_MAX_NEW_TOKENS", 512),
    )
    print(output_text)
    info = _load_open6dor_reasoning_json(output_text)
    print(info)
    return info


def manip_parsing(qwen_model, processor, image_path, instruction):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": manip_parsing_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int("SOFAR_QWEN_MANIP_PARSE_MAX_NEW_TOKENS", 128),
    )
    print(output_text)
    info = _normalize_object_direction_info(_load_qwen_json(output_text))
    print(info)
    return info


def manip_spatial_reasoning(qwen_model, processor, image_path, instruction, scene_graph):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": manip_reasoning_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Command: {instruction}\nScene Graph: {json.dumps(scene_graph, indent=2)}"},
            ],
        },
    ]
    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int("SOFAR_QWEN_MANIP_REASON_MAX_NEW_TOKENS", 256),
    )
    print(output_text)
    info = _load_manip_reasoning_json(output_text)
    info["target_orientation"] = _normalize_target_orientation(info.get("target_orientation", {}))
    try:
        info["interact_object_id"] = int(info.get("interact_object_id", 1))
    except Exception:
        info["interact_object_id"] = 1
    print(info)
    return info


def vqa_parsing(qwen_model, processor, image_path, instruction):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": vqa_parsing_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int("SOFAR_QWEN_VQA_PARSE_MAX_NEW_TOKENS", 128),
    )
    print(output_text)
    info = _normalize_object_direction_info(_load_qwen_json(output_text))
    print(info)
    return info


def vqa_spatial_reasoning(qwen_model, processor, image_path, instruction, scene_graph, eval=False):
    system_prompt = vqa_reasoning_prompt
    if eval:
        system_prompt += (
            "You will receive an image from the user, a question, and four options. "
            "You only need to respond with A, B, C, or D without providing any additional information."
        )
    else:
        system_prompt += "Provide a detail analysis of the question and the image to answer the question."

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": f"Question: {instruction}\nScene graph: {json.dumps(scene_graph, indent=2)}\nAnswer:",
                },
            ],
        },
    ]
    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int(
            "SOFAR_QWEN_VQA_EVAL_MAX_NEW_TOKENS" if eval else "SOFAR_QWEN_VQA_REASON_MAX_NEW_TOKENS",
            8 if eval else 256,
        ),
    )
    print(output_text)
    if eval:
        return _extract_choice_letter(output_text)
    return output_text
