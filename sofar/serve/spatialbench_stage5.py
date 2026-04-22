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


def _normalize_options(options: Optional[Iterable[str]]) -> List[str]:
    if not options:
        return []
    return [_normalize_text(item) for item in options if str(item or "").strip()]


def _looks_directional_option(option_text: str) -> bool:
    text = _normalize_text(option_text)
    plain = re.sub(r"[^a-z\s]", "", text).strip()
    directional_phrases = {
        "left",
        "right",
        "front",
        "back",
        "behind",
        "up",
        "down",
        "upward",
        "downward",
        "toward the camera",
        "towards the camera",
        "away from the camera",
        "to the left",
        "to the right",
        "the left",
        "the right",
        "the front",
        "the back",
    }
    if text in directional_phrases or plain in directional_phrases:
        return True
    return any(plain.startswith(prefix) for prefix in ("left", "right", "front", "back", "behind"))


def _looks_camera_alignment_option(option_text: str) -> bool:
    text = _normalize_text(option_text)
    plain = re.sub(r"[^a-z\s]", "", text).strip()
    if plain in {"yes", "no", "partially", "cannot determine"}:
        return True
    camera_markers = [
        "facing the camera",
        "facing away from the camera",
        "partially",
        "camera directly",
        "away from the camera",
    ]
    return any(marker in text or marker in plain for marker in camera_markers)


def _looks_boolean_or_mcq_option(option_text: str) -> bool:
    text = _normalize_text(option_text)
    plain = re.sub(r"[^a-z\s]", "", text).strip()
    if plain in {"yes", "no", "partially", "cannot determine"}:
        return True
    prefixes = ("yes", "no", "partially", "cannot determine")
    return any(plain.startswith(prefix) for prefix in prefixes)


def classify_spatialbench_stage5_applicability(
    question: str,
    parser_info: Optional[Dict[str, Any]] = None,
    options: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    text = _normalize_text(question)
    parser_object_count = _object_count(parser_info)
    option_texts = _normalize_options(options)
    directional_option_count = sum(_looks_directional_option(item) for item in option_texts)
    camera_alignment_option_count = sum(_looks_camera_alignment_option(item) for item in option_texts)
    boolean_option_count = sum(_looks_boolean_or_mcq_option(item) for item in option_texts)

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
    target_selection_patterns = [
        "what object is",
        "which object is",
        "which two ",
        "wearing",
        "same direction",
    ]
    axis_direction_patterns = [
        "which direction",
        "leaning to",
        "leans to",
        "point to",
        "points to",
    ]
    reference_alignment_patterns = [
        "in which direction",
        "what is the",
        "what is ",
        "facing away from",
        "facing the",
        "facing ",
        "pointing at",
        "pointing to",
        "point at",
        "point to",
    ]

    if any(pattern in text for pattern in count_patterns):
        return {
            "applicable": False,
            "category": "count_or_quantity",
            "reason": "Question is about counting / quantity rather than a single-object orientation vector.",
            "parser_object_count": parser_object_count,
            "prompt_variant": "generic",
        }

    if any(pattern in text for pattern in angle_patterns):
        return {
            "applicable": False,
            "category": "angle_or_relation",
            "reason": "Question asks for angle / parallel / opened-degree style reasoning, which is not the current head's output space.",
            "parser_object_count": parser_object_count,
            "prompt_variant": "generic",
        }

    if any(pattern in text for pattern in route_patterns):
        return {
            "applicable": False,
            "category": "route_or_navigation",
            "reason": "Question is about route / turning / navigation semantics, not a single target object's orientation vector.",
            "parser_object_count": parser_object_count,
            "prompt_variant": "generic",
        }

    if any(pattern in text for pattern in target_selection_patterns):
        return {
            "applicable": False,
            "category": "target_selection_or_relation",
            "reason": "Question uses orientation mainly to identify a target or relation, not to answer a single target object's pose directly.",
            "parser_object_count": parser_object_count,
            "prompt_variant": "generic",
        }

    if "camera" in text and camera_alignment_option_count >= max(1, min(2, len(option_texts))):
        return {
            "applicable": True,
            "category": "single_object_camera_alignment",
            "reason": "Question asks whether one target object faces the camera directly, away, or partially.",
            "parser_object_count": parser_object_count,
            "prompt_variant": "camera_alignment",
        }

    if any(pattern in text for pattern in axis_direction_patterns) and (
        directional_option_count >= 2 or (not option_texts and "which direction" in text)
    ):
        return {
            "applicable": True,
            "category": "single_object_axis_direction",
            "reason": "Question asks for a target object's dominant facing / pointing direction along coarse axes.",
            "parser_object_count": parser_object_count,
            "prompt_variant": "axis_direction",
        }

    if any(pattern in text for pattern in reference_alignment_patterns) and (
        boolean_option_count >= 2 or directional_option_count < 2 or "which direction" in text
    ):
        return {
            "applicable": True,
            "category": "single_object_reference_alignment",
            "reason": "Question asks what a target object faces / points at, or whether it aligns with a reference object.",
            "parser_object_count": parser_object_count,
            "prompt_variant": "reference_alignment",
        }

    return {
        "applicable": False,
        "category": "unsupported_orientation_semantics",
        "reason": "Question is orientation-related, but not obviously compatible with the current single-object direction head.",
        "parser_object_count": parser_object_count,
        "prompt_variant": "generic",
    }


def _axis_label(axis: str, sign: float) -> str:
    if axis == "x":
        # SpatialBench Stage 4 caches currently use the image-horizontal sign
        # opposite to the wording expected by the benchmark's left/right answers.
        return "left" if sign >= 0 else "right"
    if axis == "y":
        return "upward" if sign >= 0 else "downward"
    return "back" if sign >= 0 else "front"


def _axis_family(axis: str) -> str:
    if axis == "x":
        return "image horizontal axis"
    if axis == "y":
        return "image vertical axis"
    return "depth axis"


def _canonical_direction_from_option(option_text: str) -> Optional[str]:
    text = _normalize_text(option_text)
    plain = re.sub(r"[^a-z\s]", "", text).strip()
    mapping = {
        "left": {"left", "the left", "to the left"},
        "right": {"right", "the right", "to the right"},
        "front": {"front", "the front", "toward the camera", "towards the camera"},
        "back": {"back", "the back", "behind", "away from the camera", "farther from the camera"},
    }
    for canonical, aliases in mapping.items():
        if text in aliases or plain in aliases:
            return canonical
    return None


def _direction_score_from_vector(canonical_direction: str, x: float, z: float) -> float:
    if canonical_direction == "left":
        return x
    if canonical_direction == "right":
        return -x
    if canonical_direction == "front":
        return -z
    if canonical_direction == "back":
        return z
    return 0.0


def summarize_axis_direction_options(
    options: Optional[Iterable[str]],
    direction_vector: Iterable[float],
) -> Dict[str, Any]:
    option_list = [str(item).strip() for item in (options or []) if str(item or "").strip()]
    vec = np.asarray(list(direction_vector), dtype=np.float32).reshape(-1)
    if vec.size < 3:
        vec = np.pad(vec, (0, 3 - vec.size))
    x, _, z = [float(v) for v in vec[:3]]

    ranked_options: List[Dict[str, Any]] = []
    for index, option in enumerate(option_list):
        canonical_direction = _canonical_direction_from_option(option)
        if not canonical_direction:
            continue
        ranked_options.append(
            {
                "index": index,
                "letter": chr(ord("A") + index),
                "option": option,
                "canonical_direction": canonical_direction,
                "score": _direction_score_from_vector(canonical_direction, x, z),
            }
        )

    ranked_options.sort(key=lambda item: item["score"], reverse=True)
    preferred = ranked_options[0] if ranked_options else None
    ranking_text = ", ".join(
        f"{item['letter']}. {item['option']} ({item['canonical_direction']}, score={item['score']:.3f})"
        for item in ranked_options
    )
    score_margin = 0.0
    if len(ranked_options) >= 2:
        score_margin = float(ranked_options[0]["score"] - ranked_options[1]["score"])
    elif ranked_options:
        score_margin = abs(float(ranked_options[0]["score"]))

    return {
        "ranked_options": ranked_options,
        "ranking_text": ranking_text,
        "preferred_option": preferred["option"] if preferred else "",
        "preferred_option_letter": preferred["letter"] if preferred else "",
        "preferred_option_direction": preferred["canonical_direction"] if preferred else "",
        "preferred_option_score": float(preferred["score"]) if preferred else 0.0,
        "preferred_option_margin": score_margin,
    }


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
    options: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not prediction or not gate.get("applicable"):
        return None

    category = str(gate.get("category") or "").strip()
    prompt_variant = str(gate.get("prompt_variant") or "generic").strip() or "generic"
    description = describe_direction_vector(prediction.get("direction_vector", [0.0, 0.0, 1.0]))
    attrs = prediction.get("direction_attributes") or []
    attrs = [str(item).strip() for item in attrs if str(item or "").strip()]
    target_object = str(prediction.get("target_object") or "").strip()
    functional_part = str(prediction.get("functional_part") or "").strip()
    orientation_mode = str(prediction.get("orientation_mode") or "").strip()
    option_list = [str(item).strip() for item in (options or []) if str(item or "").strip()]
    direction_vector = prediction.get("direction_vector") or [0.0, 0.0, 1.0]
    vector_arr = np.asarray(list(direction_vector), dtype=np.float32).reshape(-1)
    if vector_arr.size < 3:
        vector_arr = np.pad(vector_arr, (0, 3 - vector_arr.size))
    z_value = float(vector_arr[2])
    axis_option_summary = summarize_axis_direction_options(option_list, vector_arr)

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
    if option_list:
        lines.append(f"options: {', '.join(option_list)}")

    reasoning_focus = "Use the Stage 5 direction evidence only when it matches the question's orientation target."
    if prompt_variant == "axis_direction":
        lines.append("direction decoding hint: +x means left, -x means right, +y means upward, -y means downward, +z means back / away from the camera, -z means front / toward the camera.")
        if axis_option_summary.get("ranking_text"):
            lines.append(f"direction option ranking: {axis_option_summary['ranking_text']}")
        if axis_option_summary.get("preferred_option_letter"):
            lines.append(
                "stage5 preferred option: "
                f"{axis_option_summary['preferred_option_letter']}. {axis_option_summary['preferred_option']}"
            )
        reasoning_focus = (
            "Map the Stage 5 vector to the closest directional option first. "
            "Prefer the top-ranked option unless the image clearly contradicts it."
        )
    elif prompt_variant == "reference_alignment":
        lines.append("reference alignment hint: combine the target object's direction evidence with scene graph object positions to decide which candidate lies along the target's facing / pointing direction.")
        reasoning_focus = "Treat Stage 5 as orientation evidence, then use the scene graph to choose the most aligned reference object or direct yes/no judgement."
    elif prompt_variant == "camera_alignment":
        if z_value <= -0.35:
            camera_hint = "the target is likely facing toward the camera"
        elif z_value >= 0.35:
            camera_hint = "the target is likely facing away from the camera"
        else:
            camera_hint = "the target is likely sideways or only partially aligned with the camera"
        lines.append(f"camera alignment hint: {camera_hint}")
        reasoning_focus = "Use the Stage 5 direction vector to judge whether the target faces the camera directly, away from it, or only partially."

    return {
        "applicable": True,
        "category": category,
        "reason": gate.get("reason"),
        "prompt_variant": prompt_variant,
        "target_object": target_object,
        "functional_part": functional_part,
        "direction_attributes": attrs,
        "direction_vector": prediction.get("direction_vector"),
        "readable_summary": description["summary"],
        "axis_hint": description["axis_hint"],
        "reasoning_focus": reasoning_focus,
        "evidence_lines": lines,
        "axis_option_ranking": axis_option_summary.get("ranked_options"),
        "preferred_option": axis_option_summary.get("preferred_option"),
        "preferred_option_letter": axis_option_summary.get("preferred_option_letter"),
        "preferred_option_score": axis_option_summary.get("preferred_option_score"),
        "preferred_option_margin": axis_option_summary.get("preferred_option_margin"),
    }
