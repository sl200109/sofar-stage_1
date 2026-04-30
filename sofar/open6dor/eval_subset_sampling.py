from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

UPRIGHT_MODES = {
    "upright",
    "upright_lens_forth",
    "upright_textual",
    "vertical",
    "watch_upright",
    "tape_measure_upright",
}
FLAT_MODES = {
    "upside_down",
    "upside_down_textual",
    "lying_flat",
    "lying_sideways",
    "lower_rim",
}
PLUG_RIGHT_MODES = {
    "plug_right",
    "plug_left",
    "handle_right",
    "handle_left",
    "handle_right_jaw_left",
    "blade_right",
    "blades_right",
    "bulb_right_handle_left",
    "prong_right",
    "spout_right",
    "ballpoint_right",
    "clasp_right",
}
CAP_CLIP_SIDEWAYS_MODES = {
    "cap_left_bottom_right",
    "cap_right_bottom_left",
    "cap_right",
    "cap_forth",
    "clip_sideways",
    "sideways",
    "sideways_textual",
    "hat_sideways",
    "remote_control_forth",
    "multimeter_forth",
    "card_forth_textual",
    "earpiece_far",
}

MODE_SUFFIX_RE = re.compile(r"(?:\.__|__)([a-z0-9_]+)$", re.IGNORECASE)


def normalize_orientation_mode(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def parse_orientation_mode_from_task_dir(task_dir: Path | str) -> str:
    path = Path(task_dir)
    for part in reversed(path.parts):
        match = MODE_SUFFIX_RE.search(part)
        if match:
            return normalize_orientation_mode(match.group(1))
        if ".__" in part:
            return normalize_orientation_mode(part.rsplit(".__", 1)[-1])
        if "__" in part:
            return normalize_orientation_mode(part.rsplit("__", 1)[-1])
    return ""


def classify_task_family(orientation_mode: Any) -> str:
    mode = normalize_orientation_mode(orientation_mode)
    if mode in UPRIGHT_MODES:
        return "upright_vertical"
    if mode in FLAT_MODES:
        return "flat_upside_down_lying_flat"
    if mode in PLUG_RIGHT_MODES:
        return "plug_right"
    if mode in CAP_CLIP_SIDEWAYS_MODES:
        return "cap_clip_sideways"
    if "upright" in mode or mode.endswith("_upright"):
        return "upright_vertical"
    if "upside_down" in mode or mode.startswith("lying_"):
        return "flat_upside_down_lying_flat"
    if any(token in mode for token in ("plug_", "handle_", "blade_", "prong_", "spout_", "ballpoint_", "clasp_")):
        return "plug_right"
    if any(token in mode for token in ("cap_", "clip_", "sideways", "_forth", "_far")):
        return "cap_clip_sideways"
    return "other"


def _stable_id(sample_key: str) -> str:
    return hashlib.md5(sample_key.encode("utf-8")).hexdigest()[:12]


def _relative_task_dir(task_dir: Path, dataset_root: Path) -> str:
    try:
        return task_dir.resolve().relative_to(dataset_root.resolve()).as_posix()
    except Exception:
        return task_dir.as_posix()


def discover_open6dor_task_dirs(dataset_root: Path) -> List[Path]:
    config_files = sorted(Path(dataset_root).rglob("task_config_new5.json"))
    task_dirs: List[Path] = []
    for config_file in config_files:
        path_str = config_file.as_posix()
        if "/task_refine_pos/" not in path_str and "/task_refine_rot/" not in path_str and "/task_refine_6dof/" not in path_str:
            continue
        task_dirs.append(config_file.parent)
    return sorted({task_dir.resolve() for task_dir in task_dirs})


def build_eval_subset_from_task_dirs(
    task_dirs: Sequence[Path],
    *,
    dataset_root: Path,
    seed: int = 42,
    target_total: int = 400,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    enriched: List[Dict[str, Any]] = []
    for task_dir in task_dirs:
        task_dir = Path(task_dir).resolve()
        relative_task_dir = _relative_task_dir(task_dir, dataset_root)
        orientation_mode = parse_orientation_mode_from_task_dir(task_dir)
        family = classify_task_family(orientation_mode)
        enriched.append(
            {
                "task_dir": task_dir.as_posix(),
                "task_dir_rel": relative_task_dir,
                "sample_id": _stable_id(relative_task_dir),
                "orientation_mode": orientation_mode,
                "task_family": family,
                "sampling_seed": seed,
                "source_dataset": "open6dor_v2",
                "source_dataset_root": dataset_root.as_posix(),
            }
        )

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        buckets[row["task_family"]].append(row)

    quotas = {
        "upright_vertical": 100,
        "flat_upside_down_lying_flat": 100,
        "plug_right": 100,
        "cap_clip_sideways": 100,
    }

    selected: List[Dict[str, Any]] = []
    selected_keys: set[str] = set()
    selection_source_counts = Counter()

    def add_rows(rows_to_add: Iterable[Dict[str, Any]], selection_source: str) -> None:
        for row in rows_to_add:
            key = row["task_dir"]
            if key in selected_keys:
                continue
            enriched_row = dict(row)
            enriched_row["selection_source"] = selection_source
            selected.append(enriched_row)
            selected_keys.add(key)
            selection_source_counts[selection_source] += 1

    for family in ["upright_vertical", "flat_upside_down_lying_flat", "plug_right", "cap_clip_sideways"]:
        family_rows = list(buckets.get(family, []))
        rng.shuffle(family_rows)
        add_rows(family_rows[: quotas[family]], "quota")

    if len(selected) < target_total:
        remainder = [row for row in enriched if row["task_dir"] not in selected_keys]
        rng.shuffle(remainder)
        add_rows(remainder[: target_total - len(selected)], "fill")

    selected.sort(key=lambda row: (row["task_family"], row["orientation_mode"], row["task_dir"]))
    if len(selected) > target_total:
        selected = selected[:target_total]
    selection_source_counts = Counter(row.get("selection_source") for row in selected)

    summary = {
        "source_dataset": "open6dor_v2",
        "source_dataset_root": dataset_root.as_posix(),
        "sampling_seed": seed,
        "target_total": target_total,
        "available_total": len(enriched),
        "empty_mode_count": sum(1 for row in enriched if not row["orientation_mode"]),
        "other_family_count": sum(1 for row in enriched if row["task_family"] == "other"),
        "selected_total": len(selected),
        "available_family_distribution": dict(Counter(row["task_family"] for row in enriched)),
        "available_orientation_mode_distribution": dict(Counter(row["orientation_mode"] for row in enriched)),
        "selected_family_distribution": dict(Counter(row["task_family"] for row in selected)),
        "selected_orientation_mode_distribution": dict(Counter(row["orientation_mode"] for row in selected)),
        "selection_source_distribution": dict(selection_source_counts),
        "quota_plan": quotas,
        "selected_sample_ids": len({row["sample_id"] for row in selected}),
    }
    return {"rows": selected, "summary": summary}


def validate_eval_subset_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    expected_family_distribution = {
        "upright_vertical": 100,
        "flat_upside_down_lying_flat": 100,
        "plug_right": 100,
        "cap_clip_sideways": 100,
    }
    selected_family_distribution = summary.get("selected_family_distribution", {}) or {}
    reasons = []
    if int(summary.get("empty_mode_count", 0)) > 0:
        reasons.append("empty_mode_count_gt_zero")
    if int(summary.get("selected_total", 0)) != 400:
        reasons.append("selected_total_not_400")
    if int(summary.get("other_family_count", 0)) > 0:
        reasons.append("other_family_present")
    if dict(selected_family_distribution) != expected_family_distribution:
        reasons.append("selected_family_distribution_mismatch")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "expected_family_distribution": expected_family_distribution,
    }


def build_eval_subset_from_dataset_root(
    dataset_root: Path,
    *,
    seed: int = 42,
    target_total: int = 400,
) -> Dict[str, Any]:
    task_dirs = discover_open6dor_task_dirs(dataset_root)
    return build_eval_subset_from_task_dirs(task_dirs, dataset_root=dataset_root, seed=seed, target_total=target_total)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
