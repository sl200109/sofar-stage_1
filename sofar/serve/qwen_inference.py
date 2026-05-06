import json
import os
import re
from qwen_vl_utils import process_vision_info
from serve.open6dor_json_utils import (
    _normalize_numeric_triplet,
    _truncate_open6dor_raw_output,
    _validate_open6dor_target_position,
    normalize_open6dor_joint_result,
    normalize_open6dor_reasoning_result,
)
from serve.system_prompts import (
    open6dor_joint_reasoning_prompt,
    open6dor_parsing_prompt,
    open6dor_reasoning_prompt,
    manip_parsing_prompt,
    manip_reasoning_prompt,
    stage2_fast_open6dor_parser_prompt,
    stage2_part_parser_prompt,
    vqa_parsing_prompt,
    vqa_reasoning_prompt,
    vqa_reasoning_stage5_axis_prompt,
    vqa_reasoning_stage5_camera_prompt,
    vqa_reasoning_stage5_prompt,
    vqa_reasoning_stage5_reference_prompt,
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

    labeled_xyz_patterns = [
        r"(?is)\bx\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*y\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*z\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"(?is)\bx-coordinate\s*(?:is|=|:)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\D+?y-coordinate\s*(?:is|=|:)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\D+?z-coordinate\s*(?:is|=|:)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]
    for pattern in labeled_xyz_patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        return {
            "calculation_process": "",
            "target_position": [float(match.group(1)), float(match.group(2)), float(match.group(3))],
        }

    parenthesized_triplets = re.findall(r"\(([^()\[\]]+)\)", cleaned)
    for candidate in reversed(parenthesized_triplets):
        triplet = _parse_numeric_triplet(candidate)
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


def _load_strict_open6dor_reasoning_json(output_text):
    cleaned = output_text.replace("```json", "").replace("```", "").strip()
    cleaned = _repair_invalid_backslash_escape(cleaned)
    obj = json.loads(cleaned)
    if isinstance(obj, dict) and "target_position" in obj:
        return obj
    if isinstance(obj, list) and len(obj) == 3:
        return {
            "calculation_process": "",
            "target_position": obj,
        }
    raise json.JSONDecodeError("Strict Open6DOR reasoning JSON schema mismatch", cleaned, 0)


def _load_strict_open6dor_joint_json(output_text):
    cleaned = output_text.replace("```json", "").replace("```", "").strip()
    cleaned = _repair_invalid_backslash_escape(cleaned)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Strict Open6DOR joint JSON schema mismatch", cleaned, 0)
    required_keys = {
        "picked_object",
        "related_objects",
        "direction_attributes",
        "target_orientation",
        "target_position",
        "calculation_process",
    }
    if not required_keys.issubset(set(obj.keys())):
        raise json.JSONDecodeError("Strict Open6DOR joint JSON missing required keys", cleaned, 0)
    return obj


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


def _normalize_object_name_text(value):
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").strip())


def _canonical_object_name(value):
    return _normalize_object_name_text(value).lower()


def _dedupe_object_names(values):
    deduped = []
    seen = set()
    for value in values:
        normalized = _normalize_object_name_text(value)
        canonical = _canonical_object_name(normalized)
        if not normalized or canonical in seen:
            continue
        deduped.append(normalized)
        seen.add(canonical)
    return deduped


def _normalize_confidence(value, default=0.0):
    try:
        score = float(value)
    except Exception:
        return default
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return round(score, 4)


def _normalize_orientation_mode_label(value):
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return ""

    canonical_map = {
        "upright": "upright",
        "standing": "upright",
        "stand": "upright",
        "vertical": "upright",
        "lying_flat": "lying_flat",
        "lie_flat": "lying_flat",
        "flat": "lying_flat",
        "lying_sideways": "lying_sideways",
        "sideways": "lying_sideways",
        "lying_sideway": "lying_sideways",
        "upside_down": "upside_down",
        "plug_right": "plug_right",
        "plug_left": "plug_left",
        "plug_up": "plug_up",
        "plug_down": "plug_down",
        "handle_right": "handle_right",
        "handle_left": "handle_left",
        "handle_up": "handle_up",
        "handle_down": "handle_down",
        "clip_sideways": "clip_sideways",
        "cap_left_bottom_right": "cap_left_bottom_right",
        "cap_right_bottom_left": "cap_right_bottom_left",
    }
    if text in canonical_map:
        return canonical_map[text]
    return text


def _extract_orientation_mode_from_text(text):
    normalized = _normalize_orientation_mode_label(text)
    if normalized:
        known_labels = [
            "cap_left_bottom_right",
            "cap_right_bottom_left",
            "clip_sideways",
            "lying_sideways",
            "lying_flat",
            "upside_down",
            "plug_right",
            "plug_left",
            "plug_up",
            "plug_down",
            "handle_right",
            "handle_left",
            "handle_up",
            "handle_down",
            "upright",
        ]
        if normalized in known_labels:
            return normalized

    lowered = str(text or "").lower()
    phrase_map = [
        ("cap left bottom right", "cap_left_bottom_right"),
        ("cap right bottom left", "cap_right_bottom_left"),
        ("clip sideways", "clip_sideways"),
        ("lying sideways", "lying_sideways"),
        ("lying flat", "lying_flat"),
        ("upside down", "upside_down"),
        ("plug right", "plug_right"),
        ("plug left", "plug_left"),
        ("plug up", "plug_up"),
        ("plug down", "plug_down"),
        ("handle right", "handle_right"),
        ("handle left", "handle_left"),
        ("handle up", "handle_up"),
        ("handle down", "handle_down"),
        ("upright", "upright"),
    ]
    for needle, label in phrase_map:
        if needle in lowered:
            return label
    return ""


def _resolve_fast_orientation_mode(model_value, task_config=None):
    task_config = task_config or {}
    task_candidates = [
        task_config.get("rot_tag_detail"),
        task_config.get("rotation_instruction"),
    ]
    for candidate in task_candidates:
        label = _extract_orientation_mode_from_text(candidate)
        if label:
            return label
    return _normalize_orientation_mode_label(model_value)


def _normalize_relation(value):
    text = str(value or "").strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    allowed = {
        "left",
        "right",
        "front",
        "behind",
        "between",
        "center",
        "count",
        "angle",
        "height",
        "distance",
        "orientation",
        "top",
        "bottom",
        "none",
    }
    if text in allowed:
        return text
    return text or "none"


def _infer_relation_from_text(text):
    lower = str(text or "").lower()
    if "between" in lower:
        return "between"
    if "center of" in lower or "at the center" in lower:
        return "center"
    if "in front of" in lower:
        return "front"
    if "behind" in lower:
        return "behind"
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    if "count" in lower or "how many" in lower:
        return "count"
    if "angle" in lower:
        return "angle"
    if "height" in lower:
        return "height"
    if "distance" in lower or "far" in lower:
        return "distance"
    if "orientation" in lower or "facing" in lower:
        return "orientation"
    return "none"


def _normalize_reference_frame(value, relation="none", functional_part="", direction_attributes=None):
    direction_attributes = direction_attributes or []
    text = str(value or "").strip().lower().replace("_", "-")
    mapping = {
        "object-centric": "object-centric",
        "object centric": "object-centric",
        "scene-centric": "scene-centric",
        "scene centric": "scene-centric",
        "world-centric": "scene-centric",
        "world centric": "scene-centric",
        "mixed": "mixed",
        "unspecified": "unspecified",
        "none": "unspecified",
    }
    if text in mapping:
        return mapping[text]

    has_part = bool(str(functional_part or "").strip()) or bool(direction_attributes)
    has_scene_relation = relation not in {"none", "orientation"}
    if has_part and has_scene_relation:
        return "mixed"
    if has_part:
        return "object-centric"
    if has_scene_relation:
        return "scene-centric"
    return "unspecified"


def _normalize_stage2_part_parser_json(raw_info, instruction=""):
    if not isinstance(raw_info, dict):
        raise ValueError(f"Unsupported Stage 2 parser schema: {raw_info}")
    info = dict(raw_info)
    direction_attributes = _ensure_string_list(info.get("direction_attributes", []))
    functional_part = str(
        info.get("functional_part")
        or info.get("part")
        or (direction_attributes[0] if direction_attributes else "")
    ).strip()
    relation = _normalize_relation(info.get("relation") or _infer_relation_from_text(instruction))
    normalized = {
        "target_object": str(
            info.get("target_object")
            or info.get("picked_object")
            or info.get("object")
            or ""
        ).strip(),
        "functional_part": functional_part,
        "relation": relation,
        "reference_object": str(
            info.get("reference_object")
            or info.get("related_object")
            or ""
        ).strip(),
        "direction_attributes": direction_attributes,
        "parser_confidence": _normalize_confidence(info.get("parser_confidence"), default=0.0),
    }
    normalized["reference_frame"] = _normalize_reference_frame(
        info.get("reference_frame"),
        relation=normalized["relation"],
        functional_part=normalized["functional_part"],
        direction_attributes=normalized["direction_attributes"],
    )
    normalized["raw_text"] = str(info.get("raw_text", "")).strip()
    return normalized


def _normalize_stage2_fast_open6dor_parser_json(raw_info, instruction="", task_config=None):
    if not isinstance(raw_info, dict):
        raise ValueError(f"Unsupported Stage 2 fast parser schema: {raw_info}")
    task_config = task_config or {}
    info = dict(raw_info)
    related_objects = _dedupe_object_names(_ensure_string_list(info.get("related_objects", [])))
    model_picked_object = _normalize_object_name_text(
        info.get("picked_object")
        or info.get("target_object")
        or ""
    )
    task_target_object = _normalize_object_name_text(task_config.get("target_obj_name", ""))
    picked_object = task_target_object or model_picked_object
    if task_target_object:
        model_picked_canonical = _canonical_object_name(model_picked_object)
        task_target_canonical = _canonical_object_name(task_target_object)
        if model_picked_object and model_picked_canonical != task_target_canonical:
            related_objects = [model_picked_object] + related_objects
        related_objects = [
            obj for obj in related_objects
            if _canonical_object_name(obj) != task_target_canonical
        ]
    related_objects = _dedupe_object_names(related_objects)
    relation = _normalize_relation(info.get("relation") or _infer_relation_from_text(instruction))
    direction_attributes = _ensure_string_list(info.get("direction_attributes", []))
    orientation_mode = _resolve_fast_orientation_mode(info.get("orientation_mode", ""), task_config=task_config)
    routing_hints = info.get("routing_hints", {}) if isinstance(info.get("routing_hints"), dict) else {}
    minimal_object_set = _dedupe_object_names(_ensure_string_list(routing_hints.get("minimal_object_set", [])))
    if not minimal_object_set:
        minimal_object_set = [picked_object] + related_objects
    else:
        minimal_object_set = [picked_object] + [
            obj for obj in minimal_object_set
            if _canonical_object_name(obj) != _canonical_object_name(picked_object)
        ]
        for related_object in related_objects:
            if _canonical_object_name(related_object) not in {
                _canonical_object_name(item) for item in minimal_object_set
            }:
                minimal_object_set.append(related_object)
    minimal_object_set = _dedupe_object_names(minimal_object_set)
    normalized = {
        "picked_object": picked_object,
        "related_objects": related_objects,
        "orientation_mode": orientation_mode,
        "relation": relation,
        "direction_attributes": direction_attributes,
        "parser_confidence": _normalize_confidence(info.get("parser_confidence"), default=0.0),
        "routing_hints": {
            "minimal_object_set": [item for item in minimal_object_set if item],
            "use_task_config": bool(routing_hints.get("use_task_config", bool(task_config))),
            "fallback_required": bool(routing_hints.get("fallback_required", not picked_object)),
        },
    }
    normalized["reference_frame"] = _normalize_reference_frame(
        info.get("reference_frame"),
        relation=normalized["relation"],
        functional_part="",
        direction_attributes=normalized["direction_attributes"],
    )
    normalized["raw_text"] = str(info.get("raw_text", "")).strip()
    return normalized


def _normalize_open6dor_joint_json(raw_info):
    if not isinstance(raw_info, dict):
        raise ValueError(f"Unsupported Open6DOR joint schema: {raw_info}")
    info = dict(raw_info)
    info["picked_object"] = str(info.get("picked_object", "")).strip()
    info["related_objects"] = _ensure_string_list(info.get("related_objects", []))
    info["direction_attributes"] = _ensure_string_list(info.get("direction_attributes", []))
    info["target_orientation"] = _normalize_target_orientation(info.get("target_orientation", {}))
    return normalize_open6dor_joint_result(info)


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


def stage2_part_parser(qwen_model, processor, image_path, instruction):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": stage2_part_parser_prompt},
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
                    "text": instruction + "\nReturn JSON only.",
                },
            ],
        },
    ]
    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int("SOFAR_QWEN_STAGE2_PART_PARSE_MAX_NEW_TOKENS", 192),
    )
    raw_info = _load_qwen_json(output_text)
    normalized = _normalize_stage2_part_parser_json(raw_info, instruction=instruction)
    normalized["raw_text"] = output_text.strip()
    print(normalized)
    return normalized


def open6dor_spatial_reasoning(
    qwen_model,
    processor,
    image_path,
    instruction,
    picked_object_info,
    other_objects_info,
    fallback_position=None,
):
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
            try:
                _load_strict_open6dor_joint_json(output_text)
                repair_applied = False
            except json.JSONDecodeError:
                repair_applied = True
            info = normalize_open6dor_reasoning_result(
                info,
                raw_output_text=output_text,
                json_repair_applied=repair_applied,
            )
            print(info)
            return info
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if strict_json:
                degraded = normalize_open6dor_reasoning_result(
                    None,
                    fallback_position=fallback_position,
                    raw_output_text=output_text,
                    json_repair_failed=True,
                    degraded_reason=exc,
                )
                print("[open6dor] reasoning JSON parse failed after strict retry, using degraded fallback result.")
                print(degraded)
                return degraded
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
    fallback_position=None,
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
            try:
                _load_strict_open6dor_reasoning_json(output_text)
                repair_applied = False
            except json.JSONDecodeError:
                repair_applied = True
            info = normalize_open6dor_joint_result(
                info,
                raw_output_text=output_text,
                json_repair_applied=repair_applied,
            )
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
                degraded = normalize_open6dor_joint_result(
                    None,
                    fallback_position=fallback_position,
                    raw_output_text=output_text,
                    json_repair_failed=True,
                    degraded_reason=exc,
                    fallback_picked_object=(object_set_hint or {}).get("picked_object", ""),
                )
                print("[open6dor] joint JSON parse failed after strict retry, using degraded fallback result.")
                print(degraded)
                return degraded
            print("[open6dor] joint JSON parse failed, retrying with stricter prompt...")

    raise last_error


def stage2_fast_open6dor_parser(qwen_model, processor, image_path, instruction, task_config=None):
    task_config = task_config or {}
    task_config_excerpt = {
        "target_obj_name": task_config.get("target_obj_name", ""),
        "instruction": task_config.get("instruction", ""),
        "position_tag": task_config.get("position_tag", ""),
        "rot_tag_detail": task_config.get("rot_tag_detail", ""),
        "rot_tag_level": task_config.get("rot_tag_level", ""),
        "rotation_instruction": task_config.get("rotation_instruction", ""),
    }
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": stage2_fast_open6dor_parser_prompt},
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
                        f"Instruction: {instruction}\n"
                        f"Task config excerpt: {json.dumps(task_config_excerpt, ensure_ascii=False)}\n"
                        "Return JSON only."
                    ),
                },
            ],
        },
    ]
    output_text = _qwen_generate_text(
        qwen_model,
        processor,
        messages,
        max_new_tokens=_env_int("SOFAR_QWEN_STAGE2_FAST_PARSE_MAX_NEW_TOKENS", 192),
    )
    raw_info = _load_qwen_json(output_text)
    normalized = _normalize_stage2_fast_open6dor_parser_json(
        raw_info,
        instruction=instruction,
        task_config=task_config_excerpt,
    )
    normalized["raw_text"] = output_text.strip()
    print(normalized)
    return normalized


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


def vqa_spatial_reasoning(qwen_model, processor, image_path, instruction, scene_graph, eval=False, stage5_context=None):
    use_stage5_prompt = bool(stage5_context and stage5_context.get("applicable"))
    if use_stage5_prompt:
        prompt_variant = str(stage5_context.get("prompt_variant") or "").strip()
        if prompt_variant == "axis_direction":
            system_prompt = vqa_reasoning_stage5_axis_prompt
        elif prompt_variant == "reference_alignment":
            system_prompt = vqa_reasoning_stage5_reference_prompt
        elif prompt_variant == "camera_alignment":
            system_prompt = vqa_reasoning_stage5_camera_prompt
        else:
            system_prompt = vqa_reasoning_stage5_prompt
    else:
        system_prompt = vqa_reasoning_prompt
    if eval:
        system_prompt += (
            "You will receive an image from the user, a question, and four options. "
            "You only need to respond with A, B, C, or D without providing any additional information."
        )
    else:
        system_prompt += "Provide a detail analysis of the question and the image to answer the question."

    stage5_block = ""
    if use_stage5_prompt:
        evidence_lines = stage5_context.get("evidence_lines") or []
        evidence_text = "\n".join(f"- {line}" for line in evidence_lines)
        reasoning_focus = str(stage5_context.get("reasoning_focus") or "").strip()
        focus_line = f"\nReasoning focus: {reasoning_focus}\n" if reasoning_focus else "\n"
        stage5_block = (
            f"\nStage 5 orientation evidence:\n{evidence_text}\n"
            f"{focus_line}"
            "Use this evidence as a high-priority cue when the question is about the target object's orientation or part direction.\n"
        )

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
                    "text": (
                        f"Question: {instruction}"
                        f"{stage5_block}\n"
                        f"Scene graph: {json.dumps(scene_graph, indent=2, ensure_ascii=False)}\n"
                        "Answer:"
                    ),
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
