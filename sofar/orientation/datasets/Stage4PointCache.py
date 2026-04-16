import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as data


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


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


def _sanitize_vector(values, size: int, default):
    arr = np.asarray(values if values is not None else default, dtype=np.float32).reshape(-1)
    if arr.size < size:
        default_arr = np.asarray(default, dtype=np.float32).reshape(-1)
        arr = np.concatenate([arr, default_arr[arr.size:size]], axis=0)
    arr = arr[:size]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if size == 3:
        norm = np.linalg.norm(arr)
        if not np.isfinite(norm) or norm <= 1e-6:
            arr = np.asarray(default, dtype=np.float32)
        else:
            arr = arr / norm
    return arr.astype(np.float32, copy=False)


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


def stage5_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "points": torch.stack([item["points"] for item in batch], dim=0),
        "target_direction": torch.stack([item["target_direction"] for item in batch], dim=0),
        "label_confidence": torch.stack([item["label_confidence"] for item in batch], dim=0),
        "prior_vector": torch.stack([item["prior_vector"] for item in batch], dim=0),
        "instruction": [item["instruction"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


class Stage4PointCacheDataset(data.Dataset):
    def __init__(self, config):
        self.manifest_path = Path(config.MANIFEST_PATH)
        self.subset = getattr(config, "subset", "all")
        self.num_points = int(getattr(config, "POINTS", 2048))
        self.max_samples = getattr(config, "MAX_SAMPLES", None)

        entries = _load_jsonl(self.manifest_path)
        if self.subset in {"train", "val", "test"}:
            entries = [item for item in entries if item.get("split", "train") == self.subset]
        if self.max_samples is not None:
            entries = entries[: int(self.max_samples)]
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        object_points = _sanitize_points(np.load(entry["object_points_path"])["points"].astype(np.float32))
        part_points_path = entry.get("part_points_path")
        part_points = None
        if part_points_path and Path(part_points_path).exists():
            part_points = _sanitize_points(np.load(part_points_path)["points"].astype(np.float32))

        points = _compose_points(object_points, part_points, self.num_points)
        target_direction = _sanitize_vector(
            entry.get("train_target_direction", entry.get("target_direction", [0.0, 0.0, 1.0])),
            3,
            [0.0, 0.0, 1.0],
        )
        prior_vector = _sanitize_vector(entry.get("prior_vector", [0.0] * 8), 8, [0.0] * 8)
        label_confidence = _sanitize_vector([entry.get("train_label_confidence", 1.0)], 1, [1.0])
        label_confidence[0] = float(np.clip(label_confidence[0], 0.0, 1.0))

        return {
            "points": torch.from_numpy(points).float(),
            "target_direction": torch.from_numpy(target_direction).float(),
            "label_confidence": torch.from_numpy(label_confidence).float(),
            "prior_vector": torch.from_numpy(prior_vector).float(),
            "instruction": str(entry.get("instruction", "")),
            "meta": {
                "dataset": entry.get("dataset"),
                "sample_key": entry.get("sample_key"),
                "target_direction_source": entry.get("train_target_direction_source", entry.get("target_direction_source")),
                "point_cache_path": entry.get("point_cache_path"),
            },
        }
