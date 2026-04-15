import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


SERVER_ROOT = "/data/coding/SoFar"


def _stable_split(sample_key: str) -> str:
    digest = hashlib.md5(sample_key.encode("utf-8")).hexdigest()
    return "val" if int(digest[:2], 16) < 51 else "train"


def _normalize_vector(vector: Optional[Iterable[float]]) -> Tuple[List[float], str]:
    if vector is None:
        return [0.0, 0.0, 1.0], "fallback_default_axis"
    arr = np.array(list(vector), dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return [0.0, 0.0, 1.0], "fallback_default_axis"
    arr = arr[:3]
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-6:
        return [0.0, 0.0, 1.0], "fallback_default_axis"
    arr = arr / norm
    return [round(float(v), 6) for v in arr], "geometry_priors.part_to_object_vector"


def _map_server_path_to_local(path_value: str, repo_root: Path) -> Path:
    path_value = str(path_value).replace("\\", "/")
    if path_value.startswith(SERVER_ROOT + "/"):
        rel = path_value[len(SERVER_ROOT) + 1 :]
        return repo_root / Path(rel)
    return Path(path_value)


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


def _build_prior_vector(cache_payload: Dict[str, Any]) -> List[float]:
    geometry = cache_payload.get("geometry_priors", {}) or {}
    grounding = cache_payload.get("grounding", {}) or {}
    vector = geometry.get("part_to_object_vector") or [0.0, 0.0, 0.0]
    vector = list(vector)[:3] if isinstance(vector, list) else [0.0, 0.0, 0.0]
    while len(vector) < 3:
        vector.append(0.0)

    object_count = float(geometry.get("object_point_count") or 0.0)
    part_count = float(geometry.get("part_point_count") or 0.0)
    part_ratio = float(geometry.get("part_ratio") or 0.0)
    object_score = float(grounding.get("object_score") or 0.0)
    part_score = float(grounding.get("part_score") or 0.0)

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


def _sample_key(dataset: str, record: Dict[str, Any]) -> str:
    if dataset == "open6dor":
        return str(record.get("task_dir", "unknown_task"))
    return str(record.get("id", record.get("sample_id", "unknown_sample")))


def _load_records(records_path: Path) -> Dict[str, Any]:
    with records_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_stage5_manifest_from_stage4_records(
    records_path: Path,
    repo_root: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    records_path = Path(records_path)
    repo_root = Path(repo_root)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_records(records_path)
    dataset = str(payload.get("dataset", "") or "unknown")
    records = payload.get("records", []) or []

    entries: List[Dict[str, Any]] = []
    skipped_missing_cache = 0
    skipped_bad_status = 0

    for record in records:
        if record.get("status") != "success":
            skipped_bad_status += 1
            continue

        cache_path_value = record.get("cache_path")
        object_points_path_value = record.get("object_points_path")
        part_points_path_value = record.get("part_points_path")
        if not cache_path_value or not object_points_path_value:
            skipped_missing_cache += 1
            continue

        cache_path = _map_server_path_to_local(cache_path_value, repo_root)
        object_points_path = _map_server_path_to_local(object_points_path_value, repo_root)
        part_points_path = (
            _map_server_path_to_local(part_points_path_value, repo_root)
            if part_points_path_value
            else None
        )
        if not cache_path.exists() or not object_points_path.exists():
            skipped_missing_cache += 1
            continue

        with cache_path.open("r", encoding="utf-8") as f:
            cache_payload = json.load(f)

        target_direction, target_source = _normalize_vector(
            (cache_payload.get("geometry_priors", {}) or {}).get("part_to_object_vector")
        )

        entry = {
            "dataset": dataset,
            "sample_key": _sample_key(dataset, record),
            "split": _stable_split(_sample_key(dataset, record)),
            "point_cache_path": str(cache_path),
            "object_points_path": str(object_points_path),
            "part_points_path": str(part_points_path) if part_points_path else None,
            "instruction": _build_instruction(dataset, cache_payload),
            "target_direction": target_direction,
            "target_direction_source": target_source,
            "prior_vector": _build_prior_vector(cache_payload),
            "parser_output": cache_payload.get("parser_output", {}) or {},
            "grounding": cache_payload.get("grounding", {}) or {},
            "geometry_priors": cache_payload.get("geometry_priors", {}) or {},
        }
        entries.append(entry)

    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "dataset": dataset,
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "total_records": len(records),
        "available_entries": len(entries),
        "skipped_missing_cache": skipped_missing_cache,
        "skipped_bad_status": skipped_bad_status,
    }
    return summary


def build_stage5_smoke_manifests(
    repo_root: Path,
    spatial_records_path: Optional[Path] = None,
    open6dor_records_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    output_dir = Path(output_dir or (repo_root / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_records_path = Path(spatial_records_path or (repo_root / "stage4_spatialbench_point_records.json"))
    open6dor_records_path = Path(open6dor_records_path or (repo_root / "stage4_open6dor_point_records.json"))

    summaries = {}
    if spatial_records_path.exists():
        summaries["spatialbench"] = build_stage5_manifest_from_stage4_records(
            spatial_records_path,
            repo_root,
            output_dir / "stage5_manifest_spatialbench_smoke.jsonl",
        )
    else:
        summaries["spatialbench"] = {
            "dataset": "spatialbench",
            "records_path": str(spatial_records_path),
            "manifest_path": str(output_dir / "stage5_manifest_spatialbench_smoke.jsonl"),
            "total_records": 0,
            "available_entries": 0,
            "skipped_missing_cache": 0,
            "skipped_bad_status": 0,
            "missing_records": True,
        }

    if open6dor_records_path.exists():
        summaries["open6dor"] = build_stage5_manifest_from_stage4_records(
            open6dor_records_path,
            repo_root,
            output_dir / "stage5_manifest_open6dor_smoke.jsonl",
        )
    else:
        summaries["open6dor"] = {
            "dataset": "open6dor",
            "records_path": str(open6dor_records_path),
            "manifest_path": str(output_dir / "stage5_manifest_open6dor_smoke.jsonl"),
            "total_records": 0,
            "available_entries": 0,
            "skipped_missing_cache": 0,
            "skipped_bad_status": 0,
            "missing_records": True,
        }

    summary_path = output_dir / "stage5_manifest_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    summaries["summary_path"] = str(summary_path)
    return summaries
