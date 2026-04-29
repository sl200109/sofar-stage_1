from __future__ import annotations

import math
from typing import Any, Dict, Optional


POLICY_VERSION = "rule_v2"

SPATIALBENCH_STAGE5_CONFIDENCE_THRESHOLD = 0.55
OPEN6DOR_STAGE5_CONFIDENCE_THRESHOLD = 0.55

SPATIALBENCH_CONTROLLER = "spatialbench_semantic_orientation_agent"
OPEN6DOR_CONTROLLER = "open6dor_semantic_orientation_agent"
AUTO_CONTROLLER = "semantic_orientation_auto_router"

OPEN6DOR_DIRECT_ALLOW_MODES = {"upright", "upright_lens_forth"}
OPEN6DOR_CONDITIONAL_VERIFY_MODES = {"plug_right", "lying_flat"}
OPEN6DOR_SEMANTIC_AXIS_VERIFY_MODES = {"lying_flat"}
OPEN6DOR_SEMANTIC_AXIS_MIN_COSINE = 0.85
OPEN6DOR_LEGACY_VERIFIER_RULE_VERSION = "rule_v2_legacy"
OPEN6DOR_SEMANTIC_AXIS_VERIFIER_RULE_VERSION = "rule_v3_semantic_axis"

SPATIALBENCH_APPLICABLE_CATEGORIES = {
    "single_object_axis_direction",
    "single_object_reference_alignment",
    "single_object_camera_alignment",
}
SPATIALBENCH_TASK_MISMATCH_CATEGORIES = {
    "count_or_quantity",
    "angle_or_relation",
    "route_or_navigation",
    "unsupported_orientation_semantics",
    "target_selection_or_relation",
}
SPATIALBENCH_TWO_OBJECT_OK_CATEGORIES = {
    "single_object_reference_alignment",
}
SPATIALBENCH_PROMPT_VARIANTS = {
    "single_object_axis_direction": "axis_direction",
    "single_object_reference_alignment": "reference_alignment",
    "single_object_camera_alignment": "camera_alignment",
}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed == float("inf") or parsed == float("-inf"):
        return default
    return parsed


def _parser_object_count(parser_info: Any) -> Optional[int]:
    if isinstance(parser_info, dict):
        return len(parser_info)
    return None


def _parser_attribute_count(parser_info: Any) -> int:
    if not isinstance(parser_info, dict):
        return 0
    total = 0
    for value in parser_info.values():
        if isinstance(value, list):
            total += len([item for item in value if str(item or "").strip()])
    return total


def _normalize_mode_label(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _spatialbench_prompt_variant(category: Any) -> str:
    return SPATIALBENCH_PROMPT_VARIANTS.get(str(category or "").strip(), "generic")


def infer_spatialbench_parser_confidence(
    parser_info: Any,
    raw_confidence: Any = None,
    stage5_applicability_category: str = "",
) -> Dict[str, Any]:
    raw = _safe_float(raw_confidence, default=None)
    if raw is not None and raw > 0.0:
        return {
            "raw_confidence": round(raw, 6),
            "effective_confidence": round(max(0.0, min(raw, 1.0)), 6),
            "source": "model",
        }

    object_count = _parser_object_count(parser_info)
    attr_count = _parser_attribute_count(parser_info)
    category = str(stage5_applicability_category or "").strip()
    if object_count == 1 and attr_count > 0:
        heuristic = 0.7
    elif object_count == 1:
        heuristic = 0.55
    elif object_count == 2 and category in SPATIALBENCH_TWO_OBJECT_OK_CATEGORIES and attr_count > 0:
        heuristic = 0.7
    elif object_count == 2 and category in SPATIALBENCH_TWO_OBJECT_OK_CATEGORIES:
        heuristic = 0.6
    elif object_count and object_count > 1:
        heuristic = 0.25
    else:
        heuristic = 0.0

    return {
        "raw_confidence": 0.0 if raw is None else round(max(0.0, min(raw, 1.0)), 6),
        "effective_confidence": round(heuristic, 6),
        "source": "heuristic",
    }


def infer_open6dor_execution_band(orientation_mode: Any) -> str:
    mode = _normalize_mode_label(orientation_mode)
    if mode in OPEN6DOR_DIRECT_ALLOW_MODES:
        return "direct_allow"
    if mode in OPEN6DOR_CONDITIONAL_VERIFY_MODES:
        return "conditional_verify"
    return "baseline_only"


def _decision_payload(
    *,
    dataset: str,
    controller: str,
    task_type: str,
    question_type_or_mode: str,
    applicable: bool,
    decision: str,
    decision_reason: str,
    selected_execution_mode: str,
    selected_actions: list[str],
    verification_status: str = "not_run",
    stage5_allowed: bool = False,
    use_orientation_prompt: bool = False,
    fallback_to_baseline: bool = True,
    agent_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "controller": controller,
        "policy_version": POLICY_VERSION,
        "task_type": task_type,
        "question_type_or_orientation_mode": question_type_or_mode,
        "applicable": bool(applicable),
        "decision": decision,
        "decision_reason": decision_reason,
        "verification_status": verification_status,
        "selected_execution_mode": selected_execution_mode,
        "selected_actions": list(selected_actions),
        "stage5_allowed": bool(stage5_allowed),
        "use_orientation_prompt": bool(use_orientation_prompt),
        "fallback_to_baseline": bool(fallback_to_baseline),
        "agent_signals": agent_signals or {},
    }


def decide_spatialbench_agent_action(
    *,
    question: str,
    task_type: str,
    question_type: str,
    options: Optional[list[str]] = None,
    parser_info: Optional[Dict[str, Any]] = None,
    parser_confidence: Any = None,
    stage5_applicability_category: str = "",
    stage5_applicability_reason: str = "",
    stage4_cache_available: bool = False,
) -> Dict[str, Any]:
    category = str(stage5_applicability_category or "").strip()
    prompt_variant = _spatialbench_prompt_variant(category)
    confidence = infer_spatialbench_parser_confidence(
        parser_info,
        raw_confidence=parser_confidence,
        stage5_applicability_category=category,
    )
    parser_object_count = _parser_object_count(parser_info)
    signals = {
        "question": question,
        "options": [str(item) for item in (options or [])],
        "parser_object_count": parser_object_count,
        "parser_confidence_raw": confidence["raw_confidence"],
        "parser_confidence": confidence["effective_confidence"],
        "parser_confidence_source": confidence["source"],
        "stage5_applicability_category": category,
        "stage5_applicability_reason": stage5_applicability_reason,
        "stage5_prompt_variant": prompt_variant,
        "stage4_cache_available": bool(stage4_cache_available),
        "stage5_prediction_available": False,
        "stage5_context_available": False,
    }

    if str(task_type or "").strip().lower() != "orientation":
        return _decision_payload(
            dataset="spatialbench",
            controller=SPATIALBENCH_CONTROLLER,
            task_type=task_type,
            question_type_or_mode=question_type,
            applicable=False,
            decision="skip_stage5_use_baseline",
            decision_reason="non_orientation_task",
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    if category in SPATIALBENCH_TASK_MISMATCH_CATEGORIES:
        return _decision_payload(
            dataset="spatialbench",
            controller=SPATIALBENCH_CONTROLLER,
            task_type=task_type,
            question_type_or_mode=question_type,
            applicable=False,
            decision="skip_stage5_due_to_task_mismatch",
            decision_reason=category or "task_mismatch",
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    if category not in SPATIALBENCH_APPLICABLE_CATEGORIES:
        return _decision_payload(
            dataset="spatialbench",
            controller=SPATIALBENCH_CONTROLLER,
            task_type=task_type,
            question_type_or_mode=question_type,
            applicable=False,
            decision="skip_stage5_due_to_task_mismatch",
            decision_reason=category or "task_mismatch",
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    allowed_object_counts = {1}
    if category in SPATIALBENCH_TWO_OBJECT_OK_CATEGORIES:
        allowed_object_counts = {1, 2}

    if parser_object_count not in allowed_object_counts:
        return _decision_payload(
            dataset="spatialbench",
            controller=SPATIALBENCH_CONTROLLER,
            task_type=task_type,
            question_type_or_mode=question_type,
            applicable=False,
            decision="skip_stage5_due_to_ambiguous_target",
            decision_reason=f"parser_object_count_not_in_{sorted(allowed_object_counts)}",
            selected_execution_mode="baseline_only",
            selected_actions=["mark_ambiguous_target_and_fallback"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    if confidence["effective_confidence"] < SPATIALBENCH_STAGE5_CONFIDENCE_THRESHOLD:
        return _decision_payload(
            dataset="spatialbench",
            controller=SPATIALBENCH_CONTROLLER,
            task_type=task_type,
            question_type_or_mode=question_type,
            applicable=False,
            decision="skip_stage5_due_to_low_confidence",
            decision_reason=f"parser_confidence_below_{SPATIALBENCH_STAGE5_CONFIDENCE_THRESHOLD:.2f}",
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    if not stage4_cache_available:
        return _decision_payload(
            dataset="spatialbench",
            controller=SPATIALBENCH_CONTROLLER,
            task_type=task_type,
            question_type_or_mode=question_type,
            applicable=False,
            decision="skip_stage5_use_baseline",
            decision_reason="stage4_cache_missing",
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    return _decision_payload(
        dataset="spatialbench",
        controller=SPATIALBENCH_CONTROLLER,
        task_type=task_type,
        question_type_or_mode=question_type,
        applicable=True,
        decision="use_stage5_with_orientation_prompt",
        decision_reason=f"{category}_ready",
        selected_execution_mode="stage5_with_orientation_prompt",
        selected_actions=[
            "run_stage5",
            "inject_scene_graph",
            "switch_orientation_prompt",
            "record_verification_outcome",
        ],
        stage5_allowed=True,
        use_orientation_prompt=True,
        fallback_to_baseline=False,
        agent_signals=signals,
    )


def verify_spatialbench_agent_outcome(
    decision: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
    stage5_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if decision.get("decision") != "use_stage5_with_orientation_prompt":
        return {
            "verification_status": "not_run",
            "verification_reason": "verification_not_required",
            "final_decision": decision.get("decision"),
            "final_decision_reason": decision.get("decision_reason"),
            "selected_execution_mode": decision.get("selected_execution_mode", "baseline_only"),
            "stage5_prediction_available": bool(prediction),
            "stage5_context_available": bool(stage5_context),
            "target_orientation_available": bool((prediction or {}).get("target_orientation")),
            "use_stage5": False,
            "use_orientation_prompt": False,
            "fallback_to_baseline": True,
            "shadow_used": False,
            "shadow_accepted": False,
            "triggered_reverification": False,
        }

    missing = []
    if not prediction or not prediction.get("direction_vector"):
        missing.append("direction_vector")
    if not prediction or not str(prediction.get("target_object") or "").strip():
        missing.append("target_object")
    if not stage5_context or not str(stage5_context.get("readable_summary") or "").strip():
        missing.append("readable_summary")

    if missing:
        return {
            "verification_status": "rejected",
            "verification_reason": "missing_stage5_evidence:" + ",".join(missing),
            "final_decision": "skip_stage5_use_baseline",
            "final_decision_reason": "missing_stage5_evidence",
            "selected_execution_mode": "fallback_reasoning",
            "stage5_prediction_available": bool(prediction),
            "stage5_context_available": bool(stage5_context),
            "target_orientation_available": bool((prediction or {}).get("target_orientation")),
            "use_stage5": False,
            "use_orientation_prompt": False,
            "fallback_to_baseline": True,
            "shadow_used": False,
            "shadow_accepted": False,
            "triggered_reverification": True,
        }

    return {
        "verification_status": "accepted",
        "verification_reason": "stage5_evidence_complete",
        "final_decision": decision.get("decision"),
        "final_decision_reason": decision.get("decision_reason"),
        "selected_execution_mode": decision.get("selected_execution_mode", "stage5_with_orientation_prompt"),
        "stage5_prediction_available": True,
        "stage5_context_available": True,
        "target_orientation_available": bool((prediction or {}).get("target_orientation")),
        "use_stage5": True,
        "use_orientation_prompt": True,
        "fallback_to_baseline": False,
        "shadow_used": False,
        "shadow_accepted": False,
        "triggered_reverification": True,
    }


def decide_open6dor_agent_action(
    *,
    stage5_enabled: bool,
    orientation_mode: str,
    stage5_gate_reason: str = "",
    fallback_required: bool = False,
    parser_confidence: Any = None,
    stage4_cache_available: bool = False,
    object_score: Any = None,
    part_score: Any = None,
    part_ratio: Any = None,
    stage5_fallback_scene_graph_used: bool = False,
    shadow_enabled: bool = False,
) -> Dict[str, Any]:
    raw_confidence = _safe_float(parser_confidence, default=None)
    mode = _normalize_mode_label(orientation_mode)
    execution_band = infer_open6dor_execution_band(mode)
    signals = {
        "orientation_mode": mode,
        "stage5_gate_reason": stage5_gate_reason,
        "fallback_required": bool(fallback_required),
        "parser_confidence": raw_confidence,
        "parser_confidence_missing": raw_confidence is None,
        "stage4_cache_available": bool(stage4_cache_available),
        "object_score": _safe_float(object_score, default=None),
        "part_score": _safe_float(part_score, default=None),
        "part_ratio": _safe_float(part_ratio, default=None),
        "stage5_fallback_scene_graph_used": bool(stage5_fallback_scene_graph_used),
        "execution_band": execution_band,
        "shadow_enabled": bool(shadow_enabled),
    }

    if not stage5_enabled:
        return _decision_payload(
            dataset="open6dor",
            controller=OPEN6DOR_CONTROLLER,
            task_type="manipulation",
            question_type_or_mode=mode,
            applicable=False,
            decision="skip_stage5_disabled",
            decision_reason="stage5_disabled",
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    if fallback_required or not stage4_cache_available:
        reason = "fallback_required" if fallback_required else "stage4_cache_missing"
        return _decision_payload(
            dataset="open6dor",
            controller=OPEN6DOR_CONTROLLER,
            task_type="manipulation",
            question_type_or_mode=mode,
            applicable=False,
            decision="skip_stage5_due_to_fallback_required",
            decision_reason=reason,
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    if raw_confidence is not None and raw_confidence < OPEN6DOR_STAGE5_CONFIDENCE_THRESHOLD:
        return _decision_payload(
            dataset="open6dor",
            controller=OPEN6DOR_CONTROLLER,
            task_type="manipulation",
            question_type_or_mode=mode,
            applicable=False,
            decision="skip_stage5_due_to_low_confidence",
            decision_reason=f"parser_confidence_below_{OPEN6DOR_STAGE5_CONFIDENCE_THRESHOLD:.2f}",
            selected_execution_mode="baseline_only",
            selected_actions=["fallback_to_baseline_reasoning"],
            stage5_allowed=False,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    if execution_band == "direct_allow":
        return _decision_payload(
            dataset="open6dor",
            controller=OPEN6DOR_CONTROLLER,
            task_type="manipulation",
            question_type_or_mode=mode,
            applicable=True,
            decision="use_stage5_direct",
            decision_reason=stage5_gate_reason or "direct_allow_mode",
            selected_execution_mode="stage5_direct_verified",
            selected_actions=[
                "run_stage5",
                "verify_stage5_orientation",
                "inject_if_verified",
            ],
            stage5_allowed=True,
            use_orientation_prompt=False,
            fallback_to_baseline=False,
            agent_signals=signals,
        )

    if execution_band == "conditional_verify":
        return _decision_payload(
            dataset="open6dor",
            controller=OPEN6DOR_CONTROLLER,
            task_type="manipulation",
            question_type_or_mode=mode,
            applicable=True,
            decision="use_stage5_conditional_verify",
            decision_reason=stage5_gate_reason or "conditional_verify_mode",
            selected_execution_mode="stage5_conditional_verified",
            selected_actions=[
                "run_stage5",
                "verify_stage5_orientation",
                "inject_if_verified",
            ],
            stage5_allowed=True,
            use_orientation_prompt=False,
            fallback_to_baseline=False,
            agent_signals=signals,
        )

    if shadow_enabled:
        return _decision_payload(
            dataset="open6dor",
            controller=OPEN6DOR_CONTROLLER,
            task_type="manipulation",
            question_type_or_mode=mode,
            applicable=False,
            decision="shadow_stage5_for_debug",
            decision_reason=stage5_gate_reason or "baseline_only_mode",
            selected_execution_mode="stage5_shadow_only",
            selected_actions=[
                "run_stage5",
                "verify_stage5_orientation",
                "record_shadow_outcome",
            ],
            stage5_allowed=True,
            use_orientation_prompt=False,
            fallback_to_baseline=True,
            agent_signals=signals,
        )

    return _decision_payload(
        dataset="open6dor",
        controller=OPEN6DOR_CONTROLLER,
        task_type="manipulation",
        question_type_or_mode=mode,
        applicable=False,
        decision="skip_stage5_due_to_mode_gating",
        decision_reason=stage5_gate_reason or "baseline_only_mode",
        selected_execution_mode="baseline_only",
        selected_actions=["fallback_to_baseline_reasoning"],
        stage5_allowed=False,
        use_orientation_prompt=False,
        fallback_to_baseline=True,
        agent_signals=signals,
    )


def _open6dor_mode_threshold(orientation_mode: str) -> Optional[tuple[str, float]]:
    mode = _normalize_mode_label(orientation_mode)
    if mode == "upright":
        return ("z_min", 0.55)
    if mode == "upright_lens_forth":
        return ("z_min", 0.35)
    if mode == "plug_right":
        return ("x_min", 0.35)
    if mode == "lying_flat":
        return ("z_abs_max", 0.45)
    return None


def _normalize_vector3(value: Any, default: Optional[list[float]] = None) -> list[float]:
    fallback = list(default or [0.0, 0.0, 1.0])
    try:
        values = list(value)
    except TypeError:
        return fallback[:3]
    if len(values) < 3:
        values = values + fallback[len(values) : 3]
    components = []
    for item in values[:3]:
        components.append(_safe_float(item, default=0.0) or 0.0)
    norm = math.sqrt(sum(component * component for component in components))
    if not math.isfinite(norm) or norm <= 1e-6:
        return fallback[:3]
    return [round(component / norm, 6) for component in components]


def _cosine_to_axis(vector: Any, axis: Any) -> float:
    vec = _normalize_vector3(vector, default=[0.0, 0.0, 1.0])
    target = _normalize_vector3(axis, default=[0.0, 0.0, 1.0])
    cosine = sum(v * t for v, t in zip(vec, target))
    if not math.isfinite(cosine):
        return 0.0
    return round(max(-1.0, min(1.0, cosine)), 6)


def _open6dor_target_axis(target_orientation: Any) -> Dict[str, Any]:
    axis = [0.0, 0.0, 1.0]
    source = "fallback_default_axis"
    label = "target_orientation"
    if isinstance(target_orientation, dict) and target_orientation:
        label, raw_axis = next(iter(target_orientation.items()))
        axis = _normalize_vector3(raw_axis, default=axis)
        source = "target_orientation"
    elif isinstance(target_orientation, (list, tuple)):
        axis = _normalize_vector3(target_orientation, default=axis)
        source = "target_orientation"
    return {
        "target_axis": axis,
        "target_axis_label": str(label or "target_orientation"),
        "target_axis_source": source,
    }


def _open6dor_verifier_debug(
    *,
    orientation_mode: str,
    vector: Any,
    target_orientation: Any,
) -> Dict[str, Any]:
    mode = _normalize_mode_label(orientation_mode)
    x = _safe_float(vector[0], default=0.0)
    y = _safe_float(vector[1], default=0.0)
    z = _safe_float(vector[2], default=0.0)
    target_axis_payload = _open6dor_target_axis(target_orientation)
    target_axis = target_axis_payload["target_axis"]
    cosine_to_target_axis = _cosine_to_axis((x, y, z), target_axis)

    threshold = _open6dor_mode_threshold(mode)
    if threshold is None:
        old_rule_pass = True
        old_rule_reason = "mode_without_additional_threshold"
    else:
        rule, value = threshold
        if rule == "z_min":
            old_rule_pass = z >= value
            old_rule_reason = f"z={z:.4f} threshold={value:.2f}"
        elif rule == "x_min":
            old_rule_pass = x >= value
            old_rule_reason = f"x={x:.4f} threshold={value:.2f}"
        else:
            old_rule_pass = abs(z) <= value
            old_rule_reason = f"|z|={abs(z):.4f} threshold={value:.2f}"

    if mode in OPEN6DOR_SEMANTIC_AXIS_VERIFY_MODES:
        new_rule_pass = cosine_to_target_axis >= OPEN6DOR_SEMANTIC_AXIS_MIN_COSINE
        verifier_rule_version = OPEN6DOR_SEMANTIC_AXIS_VERIFIER_RULE_VERSION
        verifier_decision_reason = (
            f"semantic_axis cosine={cosine_to_target_axis:.4f} "
            f"threshold={OPEN6DOR_SEMANTIC_AXIS_MIN_COSINE:.2f} "
            f"target_axis={target_axis}"
        )
    else:
        new_rule_pass = old_rule_pass
        verifier_rule_version = OPEN6DOR_LEGACY_VERIFIER_RULE_VERSION
        verifier_decision_reason = old_rule_reason

    return {
        "old_rule_pass": bool(old_rule_pass),
        "new_rule_pass": bool(new_rule_pass),
        "target_axis": target_axis,
        "target_axis_label": target_axis_payload["target_axis_label"],
        "target_axis_source": target_axis_payload["target_axis_source"],
        "cosine_to_target_axis": cosine_to_target_axis,
        "verifier_rule_version": verifier_rule_version,
        "verifier_decision_reason": verifier_decision_reason,
        "verification_reason": verifier_decision_reason,
    }


def verify_open6dor_agent_outcome(
    decision: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
    *,
    orientation_mode: str = "",
    stage4_cache_available: bool = False,
    target_orientation: Optional[Dict[str, Any]] = None,
    direction_attributes: Any = None,
) -> Dict[str, Any]:
    runnable = {
        "use_stage5_direct",
        "use_stage5_conditional_verify",
        "shadow_stage5_for_debug",
    }
    if decision.get("decision") not in runnable:
        return {
            "verification_status": "not_run",
            "verification_reason": "verification_not_required",
            "final_decision": decision.get("decision"),
            "final_decision_reason": decision.get("decision_reason"),
            "selected_execution_mode": decision.get("selected_execution_mode", "baseline_only"),
            "stage5_prediction_available": bool(prediction),
            "stage5_context_available": False,
            "target_orientation_available": bool(target_orientation),
            "use_stage5": False,
            "use_orientation_prompt": False,
            "fallback_to_baseline": True,
            "shadow_used": False,
            "shadow_accepted": False,
            "triggered_reverification": False,
        }

    if not stage4_cache_available:
        return {
            "verification_status": "rejected",
            "verification_reason": "stage4_cache_missing",
            "final_decision": "reject_stage5_keep_parser_orientation",
            "final_decision_reason": "stage4_cache_missing",
            "selected_execution_mode": "fallback_reasoning",
            "stage5_prediction_available": bool(prediction),
            "stage5_context_available": False,
            "target_orientation_available": bool(target_orientation),
            "use_stage5": False,
            "use_orientation_prompt": False,
            "fallback_to_baseline": True,
            "shadow_used": decision.get("decision") == "shadow_stage5_for_debug",
            "shadow_accepted": False,
            "triggered_reverification": True,
        }

    vector = None
    if prediction is not None:
        vector = prediction.get("direction_vector")
    if vector is None:
        return {
            "verification_status": "rejected",
            "verification_reason": "missing_stage5_evidence:direction_vector",
            "final_decision": "reject_stage5_keep_parser_orientation",
            "final_decision_reason": "missing_direction_vector",
            "selected_execution_mode": "fallback_reasoning",
            "stage5_prediction_available": bool(prediction),
            "stage5_context_available": False,
            "target_orientation_available": bool(target_orientation),
            "use_stage5": False,
            "use_orientation_prompt": False,
            "fallback_to_baseline": True,
            "shadow_used": decision.get("decision") == "shadow_stage5_for_debug",
            "shadow_accepted": False,
            "triggered_reverification": True,
            "old_rule_pass": False,
            "new_rule_pass": False,
            "target_axis": None,
            "target_axis_label": "",
            "target_axis_source": "missing",
            "cosine_to_target_axis": None,
            "verifier_rule_version": OPEN6DOR_LEGACY_VERIFIER_RULE_VERSION,
            "verifier_decision_reason": "missing_direction_vector",
        }

    target_orientation = target_orientation or (prediction or {}).get("target_orientation")
    if not target_orientation:
        return {
            "verification_status": "rejected",
            "verification_reason": "missing_stage5_evidence:target_orientation",
            "final_decision": "reject_stage5_keep_parser_orientation",
            "final_decision_reason": "missing_target_orientation",
            "selected_execution_mode": "fallback_reasoning",
            "stage5_prediction_available": bool(prediction),
            "stage5_context_available": False,
            "target_orientation_available": False,
            "use_stage5": False,
            "use_orientation_prompt": False,
            "fallback_to_baseline": True,
            "shadow_used": decision.get("decision") == "shadow_stage5_for_debug",
            "shadow_accepted": False,
            "triggered_reverification": True,
            "old_rule_pass": False,
            "new_rule_pass": False,
            "target_axis": None,
            "target_axis_label": "",
            "target_axis_source": "missing",
            "cosine_to_target_axis": None,
            "verifier_rule_version": OPEN6DOR_LEGACY_VERIFIER_RULE_VERSION,
            "verifier_decision_reason": "missing_target_orientation",
        }

    debug = _open6dor_verifier_debug(
        orientation_mode=orientation_mode,
        vector=vector,
        target_orientation=target_orientation,
    )
    old_rule_pass = bool(debug["old_rule_pass"])
    new_rule_pass = bool(debug["new_rule_pass"])
    verification_reason = str(debug["verification_reason"])
    target_axis = debug["target_axis"]
    target_axis_label = debug["target_axis_label"]
    target_axis_source = debug["target_axis_source"]
    cosine_to_target_axis = debug["cosine_to_target_axis"]
    verifier_rule_version = debug["verifier_rule_version"]
    verifier_decision_reason = debug["verifier_decision_reason"]
    accepted = new_rule_pass

    if decision.get("decision") == "shadow_stage5_for_debug":
        return {
            "verification_status": "accepted" if accepted else "rejected",
            "verification_reason": verification_reason,
            "final_decision": "shadow_stage5_for_debug",
            "final_decision_reason": decision.get("decision_reason"),
            "selected_execution_mode": "stage5_shadow_only",
            "stage5_prediction_available": True,
            "stage5_context_available": False,
            "target_orientation_available": True,
            "use_stage5": False,
            "use_orientation_prompt": False,
            "fallback_to_baseline": True,
            "shadow_used": True,
            "shadow_accepted": bool(accepted),
            "triggered_reverification": True,
            "old_rule_pass": old_rule_pass,
            "new_rule_pass": new_rule_pass,
            "target_axis": target_axis,
            "target_axis_label": target_axis_label,
            "target_axis_source": target_axis_source,
            "cosine_to_target_axis": cosine_to_target_axis,
            "verifier_rule_version": verifier_rule_version,
            "verifier_decision_reason": verifier_decision_reason,
        }

    if accepted:
        return {
            "verification_status": "accepted",
            "verification_reason": verification_reason,
            "final_decision": decision.get("decision"),
            "final_decision_reason": decision.get("decision_reason"),
            "selected_execution_mode": decision.get("selected_execution_mode"),
            "stage5_prediction_available": True,
            "stage5_context_available": False,
            "target_orientation_available": True,
            "use_stage5": True,
            "use_orientation_prompt": False,
            "fallback_to_baseline": False,
            "shadow_used": False,
            "shadow_accepted": False,
            "triggered_reverification": True,
            "old_rule_pass": old_rule_pass,
            "new_rule_pass": new_rule_pass,
            "target_axis": target_axis,
            "target_axis_label": target_axis_label,
            "target_axis_source": target_axis_source,
            "cosine_to_target_axis": cosine_to_target_axis,
            "verifier_rule_version": verifier_rule_version,
            "verifier_decision_reason": verifier_decision_reason,
        }

    return {
        "verification_status": "rejected",
        "verification_reason": verification_reason,
        "final_decision": "reject_stage5_keep_parser_orientation",
        "final_decision_reason": verification_reason,
        "selected_execution_mode": "fallback_reasoning",
        "stage5_prediction_available": True,
        "stage5_context_available": False,
        "target_orientation_available": True,
        "use_stage5": False,
        "use_orientation_prompt": False,
        "fallback_to_baseline": True,
        "shadow_used": False,
        "shadow_accepted": False,
        "triggered_reverification": True,
        "old_rule_pass": old_rule_pass,
        "new_rule_pass": new_rule_pass,
        "target_axis": target_axis,
        "target_axis_label": target_axis_label,
        "target_axis_source": target_axis_source,
        "cosine_to_target_axis": cosine_to_target_axis,
        "verifier_rule_version": verifier_rule_version,
        "verifier_decision_reason": verifier_decision_reason,
    }


def decide_auto_agent_route(
    *,
    dataset: str = "",
    question: str = "",
    task_type: str = "",
    question_type: str = "",
    task_dir: str = "",
    orientation_mode: str = "",
    child_decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    child_decision = child_decision or {}
    normalized_dataset = str(dataset or "").strip().lower()

    if normalized_dataset in {"spatialbench", "open6dor"}:
        dataset_hint = normalized_dataset
        route_reason = "explicit_dataset"
    elif question and task_type and question_type:
        dataset_hint = "spatialbench"
        route_reason = "spatialbench_schema_detected"
    elif task_dir or orientation_mode:
        dataset_hint = "open6dor"
        route_reason = "open6dor_schema_detected"
    else:
        dataset_hint = "unknown"
        route_reason = "route_unresolved"

    agent_map = {
        "spatialbench": SPATIALBENCH_CONTROLLER,
        "open6dor": OPEN6DOR_CONTROLLER,
        "unknown": "unresolved",
    }
    return {
        "controller": AUTO_CONTROLLER,
        "policy_version": POLICY_VERSION,
        "dataset_hint": dataset_hint,
        "selected_dataset_agent": agent_map[dataset_hint],
        "selected_execution_mode": child_decision.get("selected_execution_mode", "baseline_only"),
        "route_reason": route_reason,
        "child_decision": child_decision.get("decision"),
        "child_verification_status": child_decision.get("verification_status", "not_run"),
    }
