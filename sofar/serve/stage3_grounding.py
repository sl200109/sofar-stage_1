import json
from pathlib import Path

import numpy as np


def has_detections(detections):
    xyxy = getattr(detections, "xyxy", None)
    return xyxy is not None and len(xyxy) > 0


def clip_xyxy(xyxy, width, height):
    if xyxy is None or len(xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(1, min(width, int(np.ceil(x2))))
    y2 = max(1, min(height, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def first_detection_xyxy(detections, width, height):
    if not has_detections(detections):
        return None
    return clip_xyxy(detections.xyxy[0], width, height)


def first_detection_score(detections):
    confidence = getattr(detections, "confidence", None)
    if confidence is None or len(confidence) == 0:
        return None
    try:
        return float(confidence[0])
    except Exception:
        return None


def crop_image_array(image_array, xyxy):
    x1, y1, x2, y2 = xyxy
    return image_array[y1:y2, x1:x2].copy()


def expand_roi_mask(roi_mask, full_shape, roi_xyxy):
    full_mask = np.zeros(full_shape, dtype=bool)
    if roi_mask is None:
        return full_mask
    x1, y1, x2, y2 = roi_xyxy
    roi_h = max(0, y2 - y1)
    roi_w = max(0, x2 - x1)
    clipped_mask = np.array(roi_mask, dtype=bool)[:roi_h, :roi_w]
    full_mask[y1 : y1 + clipped_mask.shape[0], x1 : x1 + clipped_mask.shape[1]] = clipped_mask
    return full_mask


def constrain_child_mask_to_parent(child_mask, parent_mask):
    if child_mask is None:
        return None
    child = np.array(child_mask, dtype=bool)
    if parent_mask is None:
        return child
    parent = np.array(parent_mask, dtype=bool)
    if child.shape != parent.shape:
        raise ValueError("child_mask shape must match parent_mask shape")
    return np.logical_and(child, parent)


def offset_xyxy(roi_xyxy, parent_xyxy, full_width, full_height):
    if roi_xyxy is None:
        return None
    px1, py1, _, _ = parent_xyxy
    x1, y1, x2, y2 = roi_xyxy
    return clip_xyxy([x1 + px1, y1 + py1, x2 + px1, y2 + py1], full_width, full_height)


def serialize_xyxy(xyxy):
    if xyxy is None:
        return None
    return [int(v) for v in xyxy]


def save_stage3_cache(cache_dir, cache_payload, object_mask=None, part_mask=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "object_part_cache.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2, ensure_ascii=False)

    object_mask_path = cache_dir / "object_mask.npz"
    part_mask_path = cache_dir / "part_mask.npz"
    roi_meta_path = None

    if object_mask is not None:
        np.savez_compressed(object_mask_path, mask=np.array(object_mask, dtype=bool))
    elif object_mask_path.exists():
        object_mask_path.unlink()

    if part_mask is not None:
        if object_mask is not None:
            part_mask = constrain_child_mask_to_parent(part_mask, object_mask)
        np.savez_compressed(part_mask_path, mask=np.array(part_mask, dtype=bool))
    elif part_mask_path.exists():
        part_mask_path.unlink()

    roi_meta = cache_payload.get("roi_meta")
    if roi_meta is None:
        grounding = cache_payload.get("grounding", {})
        roi_meta = {
            "image_size": grounding.get("image_size"),
            "object_bbox_xyxy": grounding.get("object_bbox_xyxy"),
            "part_bbox_xyxy": grounding.get("part_bbox_xyxy"),
            "status": grounding.get("status"),
            "failed_stage": grounding.get("failed_stage"),
        }
    roi_meta_path = cache_dir / "roi_meta.json"
    with roi_meta_path.open("w", encoding="utf-8") as f:
        json.dump(roi_meta, f, indent=2, ensure_ascii=False)

    return {
        "cache_path": str(cache_path),
        "object_mask_path": str(object_mask_path) if object_mask_path.exists() else None,
        "part_mask_path": str(part_mask_path) if part_mask_path.exists() else None,
        "roi_meta_path": str(roi_meta_path),
    }
