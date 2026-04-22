from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def _distribution(records: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        key = str(record.get(field) or "none")
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_agent_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_records": len(records),
        "decision_distribution": _distribution(records, "agent_decision"),
        "verification_distribution": _distribution(records, "agent_verification_status"),
        "selected_execution_mode_distribution": _distribution(records, "agent_selected_execution_mode"),
        "controller_distribution": _distribution(records, "agent_controller"),
        "selected_dataset_agent_distribution": _distribution(records, "agent_selected_dataset_agent"),
        "used_stage5_count": sum(1 for item in records if item.get("agent_used_stage5")),
        "fallback_count": sum(1 for item in records if item.get("agent_fallback_to_baseline")),
        "triggered_reverification_count": sum(
            1 for item in records if item.get("agent_triggered_reverification")
        ),
        "shadow_used_count": sum(1 for item in records if item.get("agent_shadow_used")),
        "shadow_accepted_count": sum(1 for item in records if item.get("agent_shadow_accepted")),
        "rejected_count": sum(
            1 for item in records if str(item.get("agent_verification_status") or "") == "rejected"
        ),
    }


def resolve_agent_debug_dir(
    *,
    output_dir: str | Path,
    dataset: str,
    run_id: str,
    debug_root: str | Path | None = None,
) -> Path:
    root = Path(debug_root) if debug_root else Path(output_dir) / "agent_debug"
    return root / str(dataset) / str(run_id)


def write_agent_trace_bundle(
    *,
    records: List[Dict[str, Any]],
    output_dir: str | Path,
    dataset: str,
    run_id: str,
    debug_root: str | Path | None = None,
) -> Path:
    debug_dir = resolve_agent_debug_dir(
        output_dir=output_dir,
        dataset=dataset,
        run_id=run_id,
        debug_root=debug_root,
    )
    debug_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_agent_records(records)
    rejected = [
        record
        for record in records
        if str(record.get("agent_verification_status") or "") == "rejected"
        or str(record.get("agent_decision") or "") == "reject_stage5_keep_parser_orientation"
    ]
    shadow = [record for record in records if record.get("agent_shadow_used")]

    _write_jsonl(debug_dir / "agent_trace.jsonl", records)
    _write_json(
        debug_dir / "decision_summary.json",
        {
            "total_records": summary["total_records"],
            "decision_distribution": summary["decision_distribution"],
            "selected_execution_mode_distribution": summary["selected_execution_mode_distribution"],
            "controller_distribution": summary["controller_distribution"],
            "selected_dataset_agent_distribution": summary["selected_dataset_agent_distribution"],
        },
    )
    _write_json(
        debug_dir / "verification_summary.json",
        {
            "total_records": summary["total_records"],
            "verification_distribution": summary["verification_distribution"],
            "used_stage5_count": summary["used_stage5_count"],
            "fallback_count": summary["fallback_count"],
            "triggered_reverification_count": summary["triggered_reverification_count"],
            "shadow_used_count": summary["shadow_used_count"],
            "shadow_accepted_count": summary["shadow_accepted_count"],
            "rejected_count": summary["rejected_count"],
        },
    )
    _write_jsonl(debug_dir / "rejected_cases.jsonl", rejected)
    _write_jsonl(debug_dir / "shadow_cases.jsonl", shadow)
    return debug_dir


def write_agent_eval_bundle(
    *,
    output_dir: str | Path,
    dataset: str,
    mode: str,
    summary: Dict[str, Any],
    per_sample_records: List[Dict[str, Any]],
    metrics_by_bucket: Dict[str, Any],
) -> Path:
    eval_dir = Path(output_dir) / str(dataset) / str(mode)
    eval_dir.mkdir(parents=True, exist_ok=True)

    _write_json(eval_dir / "summary.json", summary)
    _write_jsonl(eval_dir / "per_sample.jsonl", per_sample_records)
    _write_json(eval_dir / "metrics_by_bucket.json", metrics_by_bucket)

    csv_path = eval_dir / "decision_breakdown.csv"
    fieldnames = [
        "sample_key",
        "task_type",
        "question_type",
        "orientation_mode",
        "bucket",
        "correct",
        "agent_controller",
        "agent_selected_dataset_agent",
        "agent_decision",
        "agent_decision_reason",
        "agent_verification_status",
        "agent_selected_execution_mode",
        "agent_used_stage5",
        "agent_fallback_to_baseline",
        "agent_shadow_used",
        "agent_shadow_accepted",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_sample_records:
            writer.writerow({field: record.get(field) for field in fieldnames})
    return eval_dir
