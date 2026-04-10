import json
from pathlib import Path

import numpy as np


def build_colored_points_from_mask(xyz_points, image_array, mask):
    xyz_points = np.array(xyz_points)
    image_array = np.array(image_array)
    mask = np.array(mask, dtype=bool)
    if xyz_points.ndim != 3 or xyz_points.shape[:2] != mask.shape:
        raise ValueError("xyz_points shape must match mask shape")
    if image_array.shape[:2] != mask.shape:
        raise ValueError("image_array shape must match mask shape")
    xyz = xyz_points[mask]
    rgb = image_array[mask]
    if xyz.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate([xyz.reshape(-1, 3), rgb.reshape(-1, 3)], axis=1).astype(np.float32, copy=False)


def sample_points(points, sample_size):
    points = np.array(points)
    if sample_size is None or sample_size <= 0 or len(points) <= sample_size:
        return points
    indices = np.random.choice(len(points), sample_size, replace=False)
    return points[indices]


def compute_geometry_priors(object_points, part_points):
    object_points = np.array(object_points)
    part_points = np.array(part_points)
    priors = {
        "object_point_count": int(len(object_points)),
        "part_point_count": int(len(part_points)),
        "part_ratio": 0.0,
        "object_centroid": None,
        "part_centroid": None,
        "part_to_object_vector": None,
    }
    if len(object_points) == 0:
        return priors

    object_xyz = object_points[:, :3]
    object_centroid = object_xyz.mean(axis=0)
    priors["object_centroid"] = [round(float(v), 4) for v in object_centroid]

    if len(part_points) > 0:
        part_xyz = part_points[:, :3]
        part_centroid = part_xyz.mean(axis=0)
        priors["part_centroid"] = [round(float(v), 4) for v in part_centroid]
        priors["part_ratio"] = round(float(len(part_points) / max(1, len(object_points))), 4)
        priors["part_to_object_vector"] = [
            round(float(v), 4) for v in (part_centroid - object_centroid)
        ]
    return priors


def save_stage4_cache(cache_dir, cache_payload, object_points, part_points=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "point_data_cache.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2, ensure_ascii=False)

    object_points_path = cache_dir / "object_points.npz"
    np.savez_compressed(object_points_path, points=np.array(object_points, dtype=np.float32))

    part_points_path = None
    if part_points is not None:
        part_points_path = cache_dir / "part_points.npz"
        np.savez_compressed(part_points_path, points=np.array(part_points, dtype=np.float32))

    return {
        "cache_path": str(cache_path),
        "object_points_path": str(object_points_path),
        "part_points_path": str(part_points_path) if part_points_path else None,
    }
