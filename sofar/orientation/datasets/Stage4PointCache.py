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


def _compose_points(object_points: np.ndarray, part_points: Optional[np.ndarray], target_count: int) -> np.ndarray:
    object_points = np.asarray(object_points, dtype=np.float32)
    part_points = np.asarray(part_points, dtype=np.float32) if part_points is not None else np.zeros((0, 6), dtype=np.float32)

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
        if self.subset in {"train", "val"}:
            entries = [item for item in entries if item.get("split", "train") == self.subset]
        if self.max_samples is not None:
            entries = entries[: int(self.max_samples)]
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        object_points = np.load(entry["object_points_path"])["points"].astype(np.float32)
        part_points_path = entry.get("part_points_path")
        part_points = None
        if part_points_path and Path(part_points_path).exists():
            part_points = np.load(part_points_path)["points"].astype(np.float32)

        points = _compose_points(object_points, part_points, self.num_points)
        target_direction = np.asarray(
            entry.get("train_target_direction", entry.get("target_direction", [0.0, 0.0, 1.0])),
            dtype=np.float32,
        )
        prior_vector = np.asarray(entry.get("prior_vector", [0.0] * 8), dtype=np.float32)
        label_confidence = np.asarray([entry.get("train_label_confidence", 1.0)], dtype=np.float32)

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
