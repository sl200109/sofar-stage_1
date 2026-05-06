from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


TARGET_FAMILIES = (
    "upright_vertical",
    "flat_upside_down_lying_flat",
    "plug_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate low-cost Open6DOR follow-up subsets from an existing subset400 run."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to open6dor_eval_subset_400_from4389_seed42.json",
    )
    parser.add_argument(
        "--records-path",
        type=Path,
        required=True,
        help="Path to stage5_open6dor_pipeline_records_*.json from the completed subset400 run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for follow-up manifests, task lists, summaries, and commands.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-core-total", type=int, default=120)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def server_repo_path(path: Path) -> str:
    relative = display_path(path)
    return f"/data/coding/SoFar/{relative}"


def normalize_records_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise TypeError("Unsupported records payload format")


def build_records_index(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in records:
        task_dir = str(row.get("task_dir") or "").strip()
        if task_dir:
            indexed[task_dir] = row
    return indexed


def annotate_rows(
    manifest_rows: Sequence[Dict[str, Any]],
    records_by_task: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for row in manifest_rows:
        record = records_by_task.get(str(row.get("task_dir")), {})
        merged = dict(row)
        merged["previous_run_error"] = record.get("error")
        merged["previous_run_mode_runtime"] = record.get("orientation_mode")
        merged["previous_run_family_runtime"] = record.get("stage5_checkpoint_family")
        merged["previous_agent_decision"] = record.get("agent_decision")
        merged["previous_selected_execution_mode"] = record.get("agent_selected_execution_mode")
        merged["previous_stage5_skip_reason"] = record.get("stage5_skip_reason")
        merged["previous_stage5_applied"] = bool(record.get("stage5_applied"))
        merged["previous_elapsed_sec"] = record.get("elapsed_sec")
        annotated.append(merged)
    return annotated


def build_error_replay_rows(
    annotated_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [row for row in annotated_rows if row.get("previous_run_error")]
    rows.sort(key=lambda row: (row.get("task_family", ""), row.get("orientation_mode", ""), row.get("task_dir", "")))
    return rows


def _round_robin_mode_sample(
    rows: Sequence[Dict[str, Any]],
    *,
    family_quota: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("orientation_mode") or "")].append(row)
    mode_names = sorted(by_mode)
    for mode_name in mode_names:
        rng.shuffle(by_mode[mode_name])
    rng.shuffle(mode_names)

    selected: List[Dict[str, Any]] = []
    selected_keys: set[str] = set()
    while len(selected) < family_quota:
        progressed = False
        for mode_name in mode_names:
            bucket = by_mode[mode_name]
            while bucket and bucket[0]["task_dir"] in selected_keys:
                bucket.pop(0)
            if not bucket:
                continue
            row = bucket.pop(0)
            selected.append(row)
            selected_keys.add(row["task_dir"])
            progressed = True
            if len(selected) >= family_quota:
                break
        if not progressed:
            break
    return selected


def build_paper_core_rows(
    annotated_rows: Sequence[Dict[str, Any]],
    *,
    seed: int,
    total_count: int,
) -> List[Dict[str, Any]]:
    if total_count % len(TARGET_FAMILIES) != 0:
        raise ValueError("paper-core total must be divisible by the number of target families")
    per_family = total_count // len(TARGET_FAMILIES)
    rng = random.Random(seed)

    eligible = [
        row
        for row in annotated_rows
        if row.get("task_family") in TARGET_FAMILIES and not row.get("previous_run_error")
    ]
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        buckets[str(row["task_family"])].append(row)

    selected: List[Dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        family_rows = list(buckets.get(family, []))
        if len(family_rows) < per_family:
            raise ValueError(f"Not enough eligible rows for family {family}: need {per_family}, got {len(family_rows)}")
        selected.extend(_round_robin_mode_sample(family_rows, family_quota=per_family, rng=rng))

    selected.sort(key=lambda row: (row.get("task_family", ""), row.get("orientation_mode", ""), row.get("task_dir", "")))
    return selected


def build_summary(
    *,
    subset_name: str,
    rows: Sequence[Dict[str, Any]],
    source_manifest_path: Path,
    source_records_path: Path,
    seed: int,
    notes: str,
) -> Dict[str, Any]:
    error_count = sum(1 for row in rows if row.get("previous_run_error"))
    execution_dist = Counter(str(row.get("previous_selected_execution_mode") or "none") for row in rows)
    decision_dist = Counter(str(row.get("previous_agent_decision") or "none") for row in rows)
    family_dist = Counter(str(row.get("task_family") or "none") for row in rows)
    mode_dist = Counter(str(row.get("orientation_mode") or "none") for row in rows)
    return {
        "subset_name": subset_name,
        "sampling_seed": seed,
        "source_manifest_path": display_path(source_manifest_path),
        "source_records_path": display_path(source_records_path),
        "selected_total": len(rows),
        "selected_error_count_from_previous_run": error_count,
        "selected_non_error_count_from_previous_run": len(rows) - error_count,
        "selected_family_distribution": dict(family_dist),
        "selected_orientation_mode_distribution": dict(mode_dist),
        "previous_selected_execution_mode_distribution": dict(execution_dist),
        "previous_agent_decision_distribution": dict(decision_dist),
        "notes": notes,
    }


def write_command(
    *,
    command_path: Path,
    task_list_path: Path,
) -> None:
    server_task_list_path = server_repo_path(task_list_path)
    command = "\n".join(
        [
            "cd /data/coding/SoFar",
            (
                "python open6dor/open6dor_perception.py "
                f"--task-list {server_task_list_path} "
                "--reset-progress --rerun-existing --use-stage5-head "
                "--stage5-checkpoint /data/coding/SoFar/output/stage5_open6dor_upright_expert_round2_semanticfix/stage5_pilot_best.pth "
                "--stage5-expert-routing task_family "
                "--stage5-upright-expert-checkpoint /data/coding/SoFar/output/stage5_open6dor_upright_expert_round2_semanticfix/stage5_pilot_best.pth "
                "--stage5-flat-expert-checkpoint /data/coding/SoFar/output/stage5_open6dor_flat_expert_round2_scratch/stage5_pilot_best.pth "
                "--stage5-plug-expert-checkpoint /data/coding/SoFar/output/stage5_open6dor_plug_expert_round2_scratch/stage5_pilot_best.pth "
                "--agent-mode dataset --agent-save-trace --agent-shadow-eval --speed-profile conservative"
            ),
        ]
    )
    command_path.write_text(command + "\n", encoding="utf-8")


def write_subset_bundle(
    *,
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    prefix: str,
    output_dir: Path,
    write_command_file: bool,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{prefix}.json"
    task_list_path = output_dir / f"{prefix}_task_list.json"
    summary_path = output_dir / f"{prefix}_summary.json"
    write_json(manifest_path, list(rows))
    write_json(task_list_path, [row["task_dir"] for row in rows])
    write_json(summary_path, summary)
    result = {
        "manifest_path": manifest_path,
        "task_list_path": task_list_path,
        "summary_path": summary_path,
    }
    if write_command_file:
        command_suffix = "_command.txt" if "error_replay" in prefix else "_final_method_command.txt"
        command_path = output_dir / f"{prefix}{command_suffix}"
        write_command(command_path=command_path, task_list_path=task_list_path)
        result["command_path"] = command_path
    return result


def main() -> None:
    args = parse_args()
    manifest_rows = load_json(args.manifest_path.resolve())
    records_payload = load_json(args.records_path.resolve())
    records = normalize_records_payload(records_payload)
    records_by_task = build_records_index(records)
    annotated_rows = annotate_rows(manifest_rows, records_by_task)

    error_replay_rows = build_error_replay_rows(annotated_rows)
    error_replay_summary = build_summary(
        subset_name="open6dor_error_replay_50_from_subset400",
        rows=error_replay_rows,
        source_manifest_path=args.manifest_path.resolve(),
        source_records_path=args.records_path.resolve(),
        seed=args.seed,
        notes="All previous subset400 error cases, used to validate targeted runtime and reasoning fixes cheaply.",
    )
    write_subset_bundle(
        rows=error_replay_rows,
        summary=error_replay_summary,
        prefix="open6dor_error_replay_50_from_subset400",
        output_dir=args.output_dir.resolve(),
        write_command_file=True,
    )

    paper_core_rows = build_paper_core_rows(
        annotated_rows,
        seed=args.seed,
        total_count=args.paper_core_total,
    )
    paper_core_summary = build_summary(
        subset_name="open6dor_paper_core_120_seed42",
        rows=paper_core_rows,
        source_manifest_path=args.manifest_path.resolve(),
        source_records_path=args.records_path.resolve(),
        seed=args.seed,
        notes="Three-family paper-core subset: 40 upright + 40 flat + 40 plug-like, excluding cap/clip and excluding previous error cases.",
    )
    paper_core_summary["quota_plan"] = {
        "upright_vertical": args.paper_core_total // 3,
        "flat_upside_down_lying_flat": args.paper_core_total // 3,
        "plug_right": args.paper_core_total // 3,
    }
    write_subset_bundle(
        rows=paper_core_rows,
        summary=paper_core_summary,
        prefix="open6dor_paper_core_120_seed42",
        output_dir=args.output_dir.resolve(),
        write_command_file=True,
    )

    print(json.dumps(
        {
            "error_replay_selected_total": len(error_replay_rows),
            "paper_core_selected_total": len(paper_core_rows),
            "paper_core_family_distribution": paper_core_summary["selected_family_distribution"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
