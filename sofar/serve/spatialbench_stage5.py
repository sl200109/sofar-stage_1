import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _object_count(parser_info: Any) -> int:
    if parser_info is None:
        return None
    if isinstance(parser_info, dict):
        return len(parser_info)
    return None


def classify_spatialbench_stage5_applicability(question: str, parser_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = _normalize_text(question)
    parser_object_count = _object_count(parser_info)

    count_patterns = [
        "how many",
        "count from",
        "count ",
        "number of",
    ]
    angle_patterns = [
        "angle difference",
        "what's the angle between",
        "what is the angle between",
        "what angle",
        "approximate angle",
        "minimum angle",
        "to what degree",
        "opened",
        "open?",
        "open ?",
        "parallel",
        "same direction",
    ]
    route_patterns = [
        "route",
        "turn",
        "walk",
        "drive",
        "intersection",
        "go downhill",
        "go deeper",
        "which direction should you",
        "according to the sign",
        "according to the blue sign",
    ]
    applicable_patterns = [
        "which direction does",
        "point to",
        "point at",
        "pointing",
        "facing",
        "leaning",
        "upright",
        "upside down",
        "tilted",
        "tilt",
    ]

    if any(pattern in text for pattern in count_patterns):
        return {
            "applicable": False,
            "category": "count_or_quantity",
            "reason": "Question is about counting / quantity rather than a single-object orientation vector.",
            "parser_object_count": parser_object_count,
        }

    if any(pattern in text for pattern in angle_patterns):
        return {
            "applicable": False,
            "category": "angle_or_relation",
            "reason": "Question asks for angle / parallel / opened-degree style reasoning, which is not the current head's output space.",
            "parser_object_count": parser_object_count,
        }

    if any(pattern in text for pattern in route_patterns):
        return {
            "applicable": False,
            "category": "route_or_navigation",
            "reason": "Question is about route / turning / navigation semantics, not a single target object's orientation vector.",
            "parser_object_count": parser_object_count,
        }

    if parser_object_count is not None and parser_object_count != 1:
        return {
            "applicable": False,
            "category": "multi_object_or_ambiguous_target",
            "reason": f"Stage 5 currently works best for a single target object, but parser returned {parser_object_count} objects.",
            "parser_object_count": parser_object_count,
        }

    if any(pattern in text for pattern in applicable_patterns):
        return {
            "applicable": True,
            "category": "single_object_direction",
            "reason": "Question focuses on one object's pointing / facing / leaning style orientation.",
            "parser_object_count": parser_object_count,
        }

    return {
        "applicable": False,
        "category": "unsupported_orientation_semantics",
        "reason": "Question is orientation-related, but not obviously compatible with the current single-object direction head.",
        "parser_object_count": parser_object_count,
    }


def _axis_label(axis: str, sign: float) -> str:
    if axis == "x":
        return "right" if sign >= 0 else "left"
    if axis == "y":
        return "upward" if sign >= 0 else "downward"
    return "farther from the camera" if sign >= 0 else "toward the camera"


def _axis_family(axis: str) -> str:
    if axis == "x":
        return "image horizontal axis"
    if axis == "y":
        return "image vertical axis"
    return "depth axis"


def describe_direction_vector(direction_vector: Iterable[float]) -> Dict[str, str]:
    vec = np.asarray(list(direction_vector), dtype=np.float32).reshape(-1)
    if vec.size < 3:
        vec = np.pad(vec, (0, 3 - vec.size))
    x, y, z = [float(v) for v in vec[:3]]
    components = [
        ("x", x, abs(x)),
        ("y", y, abs(y)),
        ("z", z, abs(z)),
    ]
    components.sort(key=lambda item: item[2], reverse=True)
    primary_axis, primary_value, primary_abs = components[0]
    secondary_axis, secondary_value, secondary_abs = components[1]

    primary_phrase = f"mostly {_axis_label(primary_axis, primary_value)}"
    secondary_phrase = ""
    if secondary_abs >= 0.25:
        modifier = "and slightly"
        if secondary_abs >= 0.5:
            modifier = "and also"
        secondary_phrase = f"{modifier} {_axis_label(secondary_axis, secondary_value)}"

    summary = primary_phrase
    if secondary_phrase:
        summary = f"{primary_phrase} {secondary_phrase}"

    if primary_abs >= 0.85:
        axis_hint = f"approximately parallel to the {_axis_family(primary_axis)}"
    elif primary_abs >= 0.55 and secondary_abs >= 0.45:
        axis_hint = (
            f"pointing {_axis_label(primary_axis, primary_value).replace('ward', '')}"
            f"-{_axis_label(secondary_axis, secondary_value).replace('ward', '')}"
        )
    else:
        axis_hint = f"dominated by the {_axis_family(primary_axis)}"

    return {
        "summary": summary,
        "axis_hint": axis_hint,
        "vector_text": f"[{x:.6f}, {y:.6f}, {z:.6f}]",
    }


def build_spatialbench_stage5_context(
    question: str,
    gate: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not prediction or not gate.get("applicable"):
        return None

    description = describe_direction_vector(prediction.get("direction_vector", [0.0, 0.0, 1.0]))
    attrs = prediction.get("direction_attributes") or []
    attrs = [str(item).strip() for item in attrs if str(item or "").strip()]
    target_object = str(prediction.get("target_object") or "").strip()
    functional_part = str(prediction.get("functional_part") or "").strip()
    orientation_mode = str(prediction.get("orientation_mode") or "").strip()

    lines: List[str] = []
    lines.append(f"target object: {target_object or 'unknown'}")
    if functional_part:
        lines.append(f"functional part: {functional_part}")
    if attrs:
        lines.append(f"direction attributes: {', '.join(attrs)}")
    if orientation_mode:
        lines.append(f"orientation mode hint: {orientation_mode}")
    lines.append(f"predicted direction vector: {description['vector_text']}")
    lines.append(f"readable summary: {description['summary']}")
    lines.append(f"axis hint: {description['axis_hint']}")

    return {
        "applicable": True,
        "category": gate.get("category"),
        "reason": gate.get("reason"),
        "target_object": target_object,
        "functional_part": functional_part,
        "direction_attributes": attrs,
        "direction_vector": prediction.get("direction_vector"),
        "readable_summary": description["summary"],
        "axis_hint": description["axis_hint"],
        "evidence_lines": lines,
    }
