import json
import re
import os
from qwen_vl_utils import process_vision_info
from serve.system_prompts import (
    open6dor_joint_reasoning_prompt,
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


def _ensure_string_list(values):
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result.append(text)
    return result


def _normalize_numeric_triplet(value):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(item) for item in value]
        except Exception:
            return None
    if isinstance(value, str):
        return _parse_numeric_triplet(value)
    return None


def _normalize_open6dor_joint_json(raw_info):
    if not isinstance(raw_info, dict):
        raise ValueError(f"Unsupported Open6DOR joint schema: {raw_info}")
    info = dict(raw_info)
    info["picked_object"] = str(info.get("picked_object", "")).strip()
    info["related_objects"] = _ensure_string_list(info.get("related_objects", []))
    info["direction_attributes"] = _ensure_string_list(info.get("direction_attributes", []))
    info["target_orientation"] = _normalize_target_orientation(info.get("target_orientation", {}))
    if not info["direction_attributes"] and info["target_orientation"]:
        info["direction_attributes"] = list(info["target_orientation"].keys())
    normalized_position = _normalize_numeric_triplet(info.get("target_position"))
    if normalized_position is None:
        raise ValueError(f"Missing valid target_position in joint result: {raw_info}")
    info["target_position"] = normalized_position
    info["calculation_process"] = str(info.get("calculation_process", "")).strip()
    return info


def _load_open6dor_joint_json(output_text):
    cleaned = output_text.replace("```json", "").replace("```", "").strip()
    cleaned = _repair_invalid_backslash_escape(cleaned)

    for candidate in _iter_json_block_candidates(cleaned):
        try:
            obj = json.loads(_repair_invalid_backslash_escape(candidate))
        except json.JSONDecodeError:
            continue
        try:
            return _normalize_open6dor_joint_json(obj)
        except Exception:
            continue

    # Last-chance fallback: reuse the reasoning extractor if the model only emitted target_position,
    # then wrap it into the joint schema with empty fields.
    info = _load_open6dor_reasoning_json(cleaned)
    return _normalize_open6dor_joint_json(
        {
            "picked_object": "",
            "related_objects": [],
            "direction_attributes": [],
            "target_orientation": {},
            "target_position": info.get("target_position", []),
            "calculation_process": info.get("calculation_process", ""),
        }
    )


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
                {
                    "type": "text",
                    "text": instruction + "\nReturn JSON only. Do not include markdown fences or natural-language explanation.",
                },
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
    def build_messages(strict_json=False):
        json_instruction = (
            'Return JSON only in the format '
            '{"calculation_process": "...", "target_position": [x, y, z]}. '
            "Do not use markdown fences."
        )
        if strict_json:
            json_instruction += (
                " Do not explain your steps outside JSON. "
                "If uncertain, still return best-effort JSON with target_position."
            )
        return [
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
                            f"{json_instruction}"
                        )
                    },
                ],
            }
        ]

    max_new_tokens = _env_int("SOFAR_QWEN_OPEN6DOR_REASON_MAX_NEW_TOKENS", 512)
    last_error = None
    for strict_json in (False, True):
        messages = build_messages(strict_json=strict_json)
        output_text = _qwen_generate_text(
            qwen_model,
            processor,
            messages,
            max_new_tokens=max_new_tokens,
        )
        print(output_text)
        try:
            info = _load_open6dor_reasoning_json(output_text)
            print(info)
            return info
        except json.JSONDecodeError as exc:
            last_error = exc
            if strict_json:
                raise
            print("[open6dor] reasoning JSON parse failed, retrying with stricter prompt...")

    raise last_error


def open6dor_joint_reasoning(
    qwen_model,
    processor,
    image_path,
    instruction,
    lightweight_scene_graph,
    object_set_hint=None,
    orientation_template_hints=None,
):
    object_set_hint = object_set_hint or {}
    orientation_template_hints = orientation_template_hints or {}

    def build_messages(strict_json=False):
        strict_clause = (
            "Return JSON only. Do not output markdown fences or natural-language explanation outside JSON."
        )
        if strict_json:
            strict_clause += (
                " Use exactly the required keys: picked_object, related_objects, direction_attributes, "
                "target_orientation, target_position, calculation_process."
            )
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": open6dor_joint_reasoning_prompt},
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
                            f"Object set hint: {json.dumps(object_set_hint, ensure_ascii=False)}\n"
                            f"Orientation template hints: {json.dumps(orientation_template_hints, ensure_ascii=False)}\n"
                            f"Lightweight scene graph: {json.dumps(lightweight_scene_graph, ensure_ascii=False)}\n"
                            f"{strict_clause}"
                        ),
                    },
                ],
            },
        ]

    max_new_tokens = _env_int("SOFAR_QWEN_OPEN6DOR_JOINT_MAX_NEW_TOKENS", 256)
    last_error = None
    for strict_json in (False, True):
        messages = build_messages(strict_json=strict_json)
        output_text = _qwen_generate_text(
            qwen_model,
            processor,
            messages,
            max_new_tokens=max_new_tokens,
        )
        print(output_text)
        try:
            info = _load_open6dor_joint_json(output_text)
            if not info["picked_object"]:
                info["picked_object"] = str(object_set_hint.get("picked_object", "")).strip()
            if not info["related_objects"]:
                info["related_objects"] = _ensure_string_list(object_set_hint.get("related_objects", []))
            if not info["direction_attributes"] and orientation_template_hints.get("direction_attributes"):
                info["direction_attributes"] = _ensure_string_list(orientation_template_hints["direction_attributes"])
            print(info)
            return info
        except Exception as exc:
            last_error = exc
            if strict_json:
                raise
            print("[open6dor] joint JSON parse failed, retrying with stricter prompt...")

    raise last_error


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
