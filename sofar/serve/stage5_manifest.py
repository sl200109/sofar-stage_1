import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


SERVER_ROOT = "/data/coding/SoFar"


def _stable_split(sample_key: str) -> str:
    digest = hashlib.md5(sample_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:2], 16)
    if bucket < 26:
        return "test"
    if bucket < 52:
        return "val"
    return "train"


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


def _normalize_named_axis(vector: Iterable[float], source: str, confidence: float) -> Tuple[List[float], str, float]:
    normalized, _ = _normalize_vector(vector)
    return normalized, source, confidence


def _resolve_pilot_label(
    dataset: str,
    cache_payload: Dict[str, Any],
) -> Tuple[List[float], str, float]:
    geometry = cache_payload.get("geometry_priors", {}) or {}
    parser_output = cache_payload.get("parser_output", {}) or {}

    vector = geometry.get("part_to_object_vector")
    if vector is not None and float(np.linalg.norm(np.array(vector, dtype=np.float32))) > 1e-6:
        normalized, source = _normalize_vector(vector)
        return normalized, source, 1.0

    if dataset == "open6dor":
        orientation_mode = str(parser_output.get("orientation_mode", "") or "").strip().lower()
        orientation_templates = {
            "upright": ([0.0, 0.0, 1.0], "orientation_mode.upright", 0.75),
            "upside_down": ([0.0, 0.0, -1.0], "orientation_mode.upside_down", 0.75),
            "lying_flat": ([1.0, 0.0, 0.0], "orientation_mode.lying_flat", 0.65),
            "lying_sideways": ([0.0, 1.0, 0.0], "orientation_mode.lying_sideways", 0.65),
            "clip_sideways": ([0.0, 1.0, 0.0], "orientation_mode.clip_sideways", 0.65),
            "plug_right": ([1.0, 0.0, 0.0], "orientation_mode.plug_right", 0.7),
            "handle_right": ([1.0, 0.0, 0.0], "orientation_mode.handle_right", 0.7),
        }
        if orientation_mode in orientation_templates:
            axis, source, confidence = orientation_templates[orientation_mode]
            return _normalize_named_axis(axis, source, confidence)

    return _normalize_named_axis([0.0, 0.0, 1.0], "fallback_default_axis", 0.2)


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


def _latest_existing_record(repo_root: Path, dataset: str) -> Optional[Path]:
    repo_root = Path(repo_root)
    filename = f"stage4_{dataset}_point_records.json"
    direct = repo_root / filename
    if direct.exists():
        return direct

    output_dir = repo_root / "output"
    output_copy = output_dir / filename
    if output_copy.exists():
        return output_copy

    candidates = sorted(output_dir.glob(f"stage4_{dataset}_point_records_*.json"))
    if candidates:
        return candidates[-1]
    return None


def _guess_cache_dir(repo_root: Path, dataset: str) -> Path:
    repo_root = Path(repo_root)
    if dataset == "spatialbench":
        return repo_root / "output" / "stage4_spatialbench_point_cache"
    return repo_root / "datasets" / "open6dor_v2"


def _scan_stage4_cache_entries(repo_root: Path, dataset: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    cache_root = _guess_cache_dir(repo_root, dataset)
    if not cache_root.exists():
        return entries

    for cache_path in cache_root.rglob("point_data_cache.json"):
        if dataset == "open6dor" and "output" not in {part.lower() for part in cache_path.parts}:
            continue

        object_points_path = cache_path.with_name("object_points.npz")
        part_points_path = cache_path.with_name("part_points.npz")
        if not object_points_path.exists():
            continue

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache_payload = json.load(f)
        except Exception:
            continue

        grounding = cache_payload.get("grounding", {}) or {}
        geometry = cache_payload.get("geometry_priors", {}) or {}
        parser_output = cache_payload.get("parser_output", {}) or {}

        if dataset == "spatialbench":
            sample_key = str(cache_payload.get("sample_id", cache_path.parent.name))
            entry = {
                "id": sample_key,
                "status": grounding.get("status", "success"),
                "cache_path": str(cache_path),
                "object_points_path": str(object_points_path),
                "part_points_path": str(part_points_path) if part_points_path.exists() else None,
                "object_point_count": geometry.get("object_point_count"),
                "part_point_count": geometry.get("part_point_count"),
                "part_ratio": geometry.get("part_ratio"),
            }
        else:
            task_dir = str(cache_path.parent.parent)
            entry = {
                "task_dir": task_dir,
                "status": grounding.get("status", "success"),
                "cache_path": str(cache_path),
                "object_points_path": str(object_points_path),
                "part_points_path": str(part_points_path) if part_points_path.exists() else None,
                "object_point_count": geometry.get("object_point_count"),
                "part_point_count": geometry.get("part_point_count"),
                "part_ratio": geometry.get("part_ratio"),
                "orientation_mode": parser_output.get("orientation_mode", ""),
            }
        entries.append(entry)

    return entries


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
        train_target_direction, train_target_source, train_label_confidence = _resolve_pilot_label(
            dataset,
            cache_payload,
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
            "train_target_direction": train_target_direction,
            "train_target_direction_source": train_target_source,
            "train_label_confidence": round(float(train_label_confidence), 4),
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


def build_stage5_manifest_from_cache_scan(
    repo_root: Path,
    dataset: str,
    manifest_path: Path,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = _scan_stage4_cache_entries(repo_root, dataset)
    payload = {"dataset": dataset, "records": records}
    temp_records = manifest_path.with_suffix(".records.tmp.json")
    with temp_records.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary = build_stage5_manifest_from_stage4_records(temp_records, repo_root, manifest_path)
    summary["records_path"] = f"cache_scan:{_guess_cache_dir(repo_root, dataset)}"
    try:
        temp_records.unlink()
    except OSError:
        pass
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

    spatial_records_path = Path(spatial_records_path) if spatial_records_path else _latest_existing_record(repo_root, "spatialbench")
    open6dor_records_path = Path(open6dor_records_path) if open6dor_records_path else _latest_existing_record(repo_root, "open6dor")

    summaries = {}
    if spatial_records_path and spatial_records_path.exists():
        summaries["spatialbench"] = build_stage5_manifest_from_stage4_records(
            spatial_records_path,
            repo_root,
            output_dir / "stage5_manifest_spatialbench_smoke.jsonl",
        )
    else:
        summaries["spatialbench"] = build_stage5_manifest_from_cache_scan(
            repo_root,
            "spatialbench",
            output_dir / "stage5_manifest_spatialbench_smoke.jsonl",
        )

    if open6dor_records_path and open6dor_records_path.exists():
        summaries["open6dor"] = build_stage5_manifest_from_stage4_records(
            open6dor_records_path,
            repo_root,
            output_dir / "stage5_manifest_open6dor_smoke.jsonl",
        )
    else:
        summaries["open6dor"] = build_stage5_manifest_from_cache_scan(
            repo_root,
            "open6dor",
            output_dir / "stage5_manifest_open6dor_smoke.jsonl",
        )

    summary_path = output_dir / "stage5_manifest_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    summaries["summary_path"] = str(summary_path)
    return summaries
