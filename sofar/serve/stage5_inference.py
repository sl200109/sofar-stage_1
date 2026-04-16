import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from serve import runtime_paths
from serve.spatialbench_stage5 import describe_direction_vector
from orientation.models.PartConditionedOrientationHead import PartConditionedOrientationHead


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _sanitize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 6)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    if points.ndim != 2:
        points = points.reshape(-1, points.shape[-1] if points.ndim > 1 else 1)
    finite_rows = np.isfinite(points).all(axis=1)
    points = points[finite_rows]
    return points.astype(np.float32, copy=False)


def _resample_points(points: np.ndarray, target_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if target_count <= 0:
        return points
    if len(points) == 0:
        return np.zeros((target_count, 6), dtype=np.float32)
    if len(points) == target_count:
        return points
    replace = len(points) < target_count
    indices = np.random.choice(len(points), size=target_count, replace=replace)
    return points[indices]


def _compose_points(object_points: np.ndarray, part_points: Optional[np.ndarray], target_count: int) -> np.ndarray:
    object_points = _sanitize_points(object_points)
    part_points = _sanitize_points(part_points) if part_points is not None else np.zeros((0, 6), dtype=np.float32)
    if len(part_points) == 0:
        return _resample_points(object_points, target_count)

    object_budget = max(target_count // 2, 1)
    part_budget = max(target_count - object_budget, 1)
    object_sample = _resample_points(object_points, object_budget)
    part_sample = _resample_points(part_points, part_budget)
    merged = np.concatenate([object_sample, part_sample], axis=0)
    if len(merged) != target_count:
        merged = _resample_points(merged, target_count)
    return merged


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _build_prior_vector(cache_payload: Dict[str, Any]) -> List[float]:
    geometry = cache_payload.get("geometry_priors", {}) or {}
    grounding = cache_payload.get("grounding", {}) or {}
    vector = geometry.get("part_to_object_vector") or [0.0, 0.0, 0.0]
    vector = list(vector)[:3] if isinstance(vector, list) else [0.0, 0.0, 0.0]
    while len(vector) < 3:
        vector.append(0.0)
    vector = [_safe_float(v, 0.0) for v in vector[:3]]

    object_count = _safe_float(geometry.get("object_point_count"), 0.0)
    part_count = _safe_float(geometry.get("part_point_count"), 0.0)
    part_ratio = _safe_float(geometry.get("part_ratio"), 0.0)
    object_score = _safe_float(grounding.get("object_score"), 0.0)
    part_score = _safe_float(grounding.get("part_score"), 0.0)

    return [
        round(part_ratio, 6),
        round(object_score, 6),
        round(part_score, 6),
        round(min(object_count / 4096.0, 1.0), 6),
        round(min(part_count / 4096.0, 1.0), 6),
        round(float(vector[0]), 6),
        round(float(vector[1]), 6),
        round(float(vector[2]), 6),
    ]


def _build_instruction(dataset: str, cache_payload: Dict[str, Any]) -> str:
    parser_output = cache_payload.get("parser_output", {}) or {}
    grounding = cache_payload.get("grounding", {}) or {}

    if dataset == "open6dor":
        picked_object = parser_output.get("picked_object") or grounding.get("target_object") or "object"
        orientation_mode = parser_output.get("orientation_mode") or "unspecified"
        related_objects = parser_output.get("related_objects") or []
        if not isinstance(related_objects, list):
            related_objects = [str(related_objects)]
        direction_attributes = parser_output.get("direction_attributes") or []
        if not isinstance(direction_attributes, list):
            direction_attributes = [str(direction_attributes)]
        reference_text = ", ".join([str(x) for x in related_objects if str(x).strip()]) or "none"
        part_text = ", ".join([str(x) for x in direction_attributes if str(x).strip()]) or "none"
        return (
            f"Target object: {picked_object}. "
            f"Orientation mode: {orientation_mode}. "
            f"Relevant parts: {part_text}. "
            f"Reference objects: {reference_text}."
        )

    target_object = parser_output.get("target_object") or grounding.get("target_object") or "object"
    functional_part = parser_output.get("functional_part") or grounding.get("functional_part") or "none"
    relation = parser_output.get("relation") or "unspecified"
    reference_object = parser_output.get("reference_object") or grounding.get("reference_object") or "none"
    return (
        f"Target object: {target_object}. "
        f"Functional part: {functional_part}. "
        f"Relation: {relation}. "
        f"Reference object: {reference_object}."
    )


def _infer_direction_attributes(dataset: str, cache_payload: Dict[str, Any]) -> List[str]:
    parser_output = cache_payload.get("parser_output", {}) or {}
    if dataset == "open6dor":
        values = parser_output.get("direction_attributes") or []
        if not isinstance(values, list):
            values = [str(values)]
        return [str(item).strip() for item in values if str(item or "").strip()]

    functional_part = str(parser_output.get("functional_part", "") or "").strip()
    if functional_part:
        return [functional_part]
    values = parser_output.get("direction_attributes") or []
    if not isinstance(values, list):
        values = [str(values)]
    return [str(item).strip() for item in values if str(item or "").strip()]


def _vector_to_orientation_dict(direction_attributes: Iterable[str], direction_vector: Iterable[float]) -> Dict[str, List[float]]:
    attrs = [str(item).strip() for item in direction_attributes if str(item or "").strip()]
    vector = [round(float(v), 6) for v in np.asarray(direction_vector, dtype=np.float32).reshape(-1)[:3]]
    return {attr: vector for attr in attrs}


def _format_direction_string(direction_vector: Iterable[float]) -> str:
    vec = np.asarray(direction_vector, dtype=np.float32).reshape(-1)
    while vec.size < 3:
        vec = np.append(vec, 0.0)
    return f"direction vector: ({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Stage5OrientationPredictor:
    def __init__(self, checkpoint_path: Path, device: str = "auto", num_points: int = 1024):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = _resolve_device(device)
        self.num_points = int(num_points)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        model_config = payload.get("model_config") or {
            "point_dim": 6,
            "prior_dim": 8,
            "hidden_dim": 128,
            "text_dim": 64,
            "vocab_size": 4096,
            "max_tokens": 32,
        }
        self.model = PartConditionedOrientationHead(SimpleNamespace(**model_config)).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()

    def predict(self, object_points, part_points, instruction: str, prior_vector: Optional[Iterable[float]] = None) -> List[float]:
        points = _compose_points(object_points, part_points, self.num_points)
        priors = np.asarray(prior_vector if prior_vector is not None else [0.0] * 8, dtype=np.float32).reshape(-1)
        if priors.size < 8:
            priors = np.pad(priors, (0, 8 - priors.size))
        priors = priors[:8]
        priors = np.nan_to_num(priors, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        with torch.no_grad():
            pts_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
            priors_tensor = torch.from_numpy(priors).float().unsqueeze(0).to(self.device)
            pred = self.model(pts_tensor, [str(instruction or "")], priors_tensor)
            pred = torch.nn.functional.normalize(pred, dim=-1)[0].detach().cpu().numpy()
        return [round(float(v), 6) for v in pred.tolist()]


@lru_cache(maxsize=8)
def _cached_predictor(checkpoint_path: str, device: str, num_points: int):
    return Stage5OrientationPredictor(Path(checkpoint_path), device=device, num_points=num_points)


def load_stage5_predictor(checkpoint_path: Path, device: str = "auto", num_points: int = 1024):
    return _cached_predictor(str(Path(checkpoint_path).resolve()), _resolve_device(device), int(num_points))


def resolve_stage5_checkpoint(dataset: str, checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path).resolve()
    dataset = str(dataset or "").strip().lower()
    if dataset == "open6dor":
        return runtime_paths.stage5_open6dor_checkpoint_path()
    if dataset == "spatialbench":
        return runtime_paths.stage5_spatialbench_checkpoint_path()
    raise ValueError(f"Unsupported dataset for stage5 checkpoint resolution: {dataset}")


def predict_from_stage4_dir(
    stage4_dir: str | Path,
    dataset: str,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    num_points: int = 1024,
) -> Optional[Dict[str, Any]]:
    stage4_dir = Path(stage4_dir)
    cache_path = stage4_dir / "point_data_cache.json"
    object_points_path = stage4_dir / "object_points.npz"
    part_points_path = stage4_dir / "part_points.npz"
    if not cache_path.exists() or not object_points_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as f:
        cache_payload = json.load(f)

    object_points = np.load(object_points_path)["points"].astype(np.float32)
    part_points = None
    if part_points_path.exists():
        part_points = np.load(part_points_path)["points"].astype(np.float32)

    predictor = load_stage5_predictor(resolve_stage5_checkpoint(dataset, checkpoint_path), device=device, num_points=num_points)
    instruction = _build_instruction(dataset, cache_payload)
    prior_vector = _build_prior_vector(cache_payload)
    direction_vector = predictor.predict(object_points, part_points, instruction, prior_vector)
    direction_attributes = _infer_direction_attributes(dataset, cache_payload)
    target_orientation = _vector_to_orientation_dict(direction_attributes, direction_vector)

    return {
        "dataset": dataset,
        "checkpoint_path": str(predictor.checkpoint_path),
        "device": predictor.device,
        "point_cache_path": str(cache_path),
        "object_points_path": str(object_points_path),
        "part_points_path": str(part_points_path) if part_points_path.exists() else None,
        "instruction": instruction,
        "prior_vector": prior_vector,
        "direction_vector": direction_vector,
        "direction_attributes": direction_attributes,
        "target_orientation": target_orientation,
        "target_object": (cache_payload.get("parser_output", {}) or {}).get("target_object")
        or (cache_payload.get("parser_output", {}) or {}).get("picked_object")
        or (cache_payload.get("grounding", {}) or {}).get("target_object")
        or (cache_payload.get("grounding", {}) or {}).get("picked_object")
        or "",
        "functional_part": (cache_payload.get("parser_output", {}) or {}).get("functional_part", ""),
        "orientation_mode": (cache_payload.get("parser_output", {}) or {}).get("orientation_mode", ""),
    }


def inject_prediction_into_scene_graph(scene_graph: List[Dict[str, Any]], prediction: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not prediction:
        return scene_graph

    target_object = _normalize_name(prediction.get("target_object"))
    vector_str = _format_direction_string(prediction.get("direction_vector", [0.0, 0.0, 1.0]))
    vector_description = describe_direction_vector(prediction.get("direction_vector", [0.0, 0.0, 1.0]))
    attrs = prediction.get("direction_attributes") or []
    attrs = [str(item).strip() for item in attrs if str(item or "").strip()]
    target_index = None

    for idx, node in enumerate(scene_graph):
        node_name = _normalize_name(node.get("object name"))
        if target_object and node_name == target_object:
            target_index = idx
            break
    if target_index is None and scene_graph:
        target_index = 0
    if target_index is None:
        return scene_graph

    target_node = dict(scene_graph[target_index])
    orientation_info = target_node.get("orientation")
    if not isinstance(orientation_info, dict):
        orientation_info = {}
    if attrs:
        for attr in attrs:
            orientation_info[attr] = vector_str
    orientation_info["stage5_summary"] = vector_description["summary"]
    orientation_info["stage5_head"] = vector_str
    target_node["orientation"] = orientation_info
    target_node["stage5_head"] = {
        "direction_vector": prediction.get("direction_vector"),
        "direction_attributes": attrs,
        "checkpoint_path": prediction.get("checkpoint_path"),
        "functional_part": prediction.get("functional_part"),
        "orientation_mode": prediction.get("orientation_mode"),
    }
    target_node["stage5_evidence"] = {
        "direction_vector": prediction.get("direction_vector"),
        "direction_attributes": attrs,
        "readable_summary": vector_description["summary"],
        "axis_hint": vector_description["axis_hint"],
        "checkpoint_path": prediction.get("checkpoint_path"),
    }
    scene_graph = list(scene_graph)
    scene_graph[target_index] = target_node
    return scene_graph
