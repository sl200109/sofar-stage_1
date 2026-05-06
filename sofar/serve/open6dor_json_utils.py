import math


def _parse_numeric_triplet(text):
    import re

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text or ""))
    if len(matches) != 3:
        return None
    return [float(value) for value in matches]


def _normalize_numeric_triplet(value):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(item) for item in value]
        except Exception:
            return None
    if isinstance(value, str):
        return _parse_numeric_triplet(value)
    return None


def _truncate_open6dor_raw_output(text, limit=400):
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[:limit].rstrip() + "...<truncated>"


def _short_open6dor_error_text(error, limit=160):
    text = str(error or "").strip() or "unknown_error"
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...<truncated>"


def _validate_open6dor_target_position(value):
    triplet = _normalize_numeric_triplet(value)
    if triplet is None:
        raise ValueError(f"Invalid target_position schema: {value}")
    if any(not math.isfinite(float(item)) for item in triplet):
        raise ValueError(f"Non-finite target_position values: {value}")
    return [float(item) for item in triplet]


def _resolve_safe_degraded_position(fallback_position):
    try:
        return _validate_open6dor_target_position(fallback_position), "fallback_position"
    except Exception:
        return [0.0, 0.0, 0.0], "zero_fallback"


def normalize_open6dor_reasoning_result(
    raw_info,
    *,
    fallback_position=None,
    raw_output_text="",
    json_repair_applied=False,
    json_repair_failed=False,
    degraded_reason="",
):
    if json_repair_failed:
        safe_position, position_source = _resolve_safe_degraded_position(fallback_position)
        return {
            "calculation_process": "degraded fallback because Open6DOR reasoning JSON parse failed",
            "target_position": safe_position,
            "json_repair_applied": False,
            "json_repair_failed": True,
            "degraded_position_source": position_source,
            "degraded_reason": _short_open6dor_error_text(degraded_reason),
            "raw_reasoning_output": _truncate_open6dor_raw_output(raw_output_text),
        }

    if isinstance(raw_info, list):
        raw_info = {
            "calculation_process": "",
            "target_position": raw_info,
        }
    if not isinstance(raw_info, dict):
        raise ValueError(f"Unsupported Open6DOR reasoning schema: {raw_info}")

    normalized = dict(raw_info)
    normalized["calculation_process"] = str(raw_info.get("calculation_process", "")).strip()
    normalized["target_position"] = _validate_open6dor_target_position(raw_info.get("target_position"))
    normalized["json_repair_applied"] = bool(json_repair_applied)
    normalized["json_repair_failed"] = False
    if json_repair_applied:
        normalized["raw_reasoning_output"] = _truncate_open6dor_raw_output(raw_output_text)
    return normalized


def normalize_open6dor_joint_result(
    raw_info,
    *,
    fallback_position=None,
    raw_output_text="",
    json_repair_applied=False,
    json_repair_failed=False,
    degraded_reason="",
    fallback_picked_object="",
):
    if json_repair_failed:
        safe_position, position_source = _resolve_safe_degraded_position(fallback_position)
        return {
            "picked_object": str(fallback_picked_object or "").strip(),
            "related_objects": [],
            "direction_attributes": [],
            "target_orientation": {},
            "calculation_process": "degraded fallback because Open6DOR joint JSON parse failed",
            "target_position": safe_position,
            "json_repair_applied": False,
            "json_repair_failed": True,
            "degraded_position_source": position_source,
            "degraded_reason": _short_open6dor_error_text(degraded_reason),
            "raw_reasoning_output": _truncate_open6dor_raw_output(raw_output_text),
        }

    if not isinstance(raw_info, dict):
        raise ValueError(f"Unsupported Open6DOR joint schema: {raw_info}")

    normalized = dict(raw_info)
    normalized["picked_object"] = str(raw_info.get("picked_object", "")).strip()
    normalized["related_objects"] = [
        str(item).strip() for item in raw_info.get("related_objects", []) if str(item).strip()
    ]
    normalized["direction_attributes"] = [
        str(item).strip() for item in raw_info.get("direction_attributes", []) if str(item).strip()
    ]
    target_orientation = raw_info.get("target_orientation", {})
    normalized["target_orientation"] = target_orientation if isinstance(target_orientation, dict) else {}
    if not normalized["direction_attributes"] and normalized["target_orientation"]:
        normalized["direction_attributes"] = list(normalized["target_orientation"].keys())
    normalized["calculation_process"] = str(raw_info.get("calculation_process", "")).strip()
    normalized["target_position"] = _validate_open6dor_target_position(raw_info.get("target_position"))
    normalized["json_repair_applied"] = bool(json_repair_applied)
    normalized["json_repair_failed"] = False
    if json_repair_applied:
        normalized["raw_reasoning_output"] = _truncate_open6dor_raw_output(raw_output_text)
    return normalized
