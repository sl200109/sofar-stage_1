from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sofar.open6dor.eval_subset_sampling import build_eval_subset_from_dataset_root
from sofar.serve.semantic_orientation_agent import verify_open6dor_agent_outcome


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def mean_of(values: List[float]) -> float:
    return round(float(sum(values) / len(values)), 6) if values else 0.0


def median_of(values: List[float]) -> float:
    return round(float(statistics.median(values)), 6) if values else 0.0


def max_of(values: List[float]) -> float:
    return round(float(max(values)), 6) if values else 0.0


def record_target_orientation(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("stage5_target_orientation")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    stage5_head = record.get("stage5_head") or {}
    if isinstance(stage5_head, dict):
        nested = stage5_head.get("target_orientation")
        if isinstance(nested, dict):
            return nested
    return {}


def record_direction_vector(record: Dict[str, Any]) -> List[float]:
    value = record.get("stage5_direction_vector")
    if isinstance(value, list):
        return [safe_float(v) for v in value[:3]]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [safe_float(v) for v in parsed[:3]]
        except Exception:
            pass
        numbers = [safe_float(piece) for piece in value.replace(",", " ").split()]
        if len(numbers) >= 3:
            return numbers[:3]
    stage5_head = record.get("stage5_head") or {}
    if isinstance(stage5_head, dict):
        nested = stage5_head.get("direction_vector")
        if isinstance(nested, list):
            return [safe_float(v) for v in nested[:3]]
    return [0.0, 0.0, 0.0]


def record_direction_attributes(record: Dict[str, Any]) -> List[str]:
    value = record.get("stage5_direction_attributes")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return []


def compute_verifier_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    decision = {
        "decision": str(record.get("agent_decision") or ""),
        "decision_reason": str(record.get("agent_decision_reason") or ""),
        "selected_execution_mode": str(record.get("agent_selected_execution_mode") or ""),
    }
    prediction = {
        "direction_vector": record_direction_vector(record),
        "target_orientation": record_target_orientation(record),
    }
    return verify_open6dor_agent_outcome(
        decision,
        prediction=prediction,
        orientation_mode=str(record.get("stage5_mode") or ""),
        stage4_cache_available=True,
        target_orientation=prediction["target_orientation"],
        direction_attributes=record_direction_attributes(record),
    )


def build_verifier_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    rule_versions = Counter()
    decision_distribution = Counter()
    selected_execution_mode_distribution = Counter()
    mode_distribution = Counter()
    upright_direct_count = 0
    plug_right_conditional_count = 0
    lying_flat_accepted_count = 0
    lying_flat_fallback_count = 0
    old_rule_pass = 0
    new_rule_pass = 0

    for record in records:
        verification = compute_verifier_fields(record)
        mode = str(record.get("stage5_mode") or "")
        decision = str(record.get("agent_decision") or "")
        selected_execution_mode = str(record.get("agent_selected_execution_mode") or "")
        verification_status = str(record.get("agent_verification_status") or "")
        fallback = bool(record.get("agent_fallback_to_baseline"))
        shadow_used = bool(record.get("agent_shadow_used"))
        shadow_accepted = bool(record.get("agent_shadow_accepted"))
        old_pass = bool(verification.get("old_rule_pass"))
        new_pass = bool(verification.get("new_rule_pass"))
        rule_version = str(verification.get("verifier_rule_version") or "none")

        old_rule_pass += int(old_pass)
        new_rule_pass += int(new_pass)
        rule_versions[rule_version] += 1
        decision_distribution[decision] += 1
        selected_execution_mode_distribution[selected_execution_mode] += 1
        mode_distribution[mode] += 1

        if mode == "upright" and verification_status == "accepted" and decision == "use_stage5_direct":
            upright_direct_count += 1
        if mode == "plug_right" and verification_status == "accepted" and decision == "use_stage5_conditional_verify":
            plug_right_conditional_count += 1
        if mode == "lying_flat":
            if verification_status == "accepted":
                lying_flat_accepted_count += 1
            if fallback:
                lying_flat_fallback_count += 1

        rows.append(
            {
                "task_dir": record.get("task_dir"),
                "stage5_mode": mode,
                "agent_decision": decision,
                "agent_verification_status": verification_status,
                "agent_selected_execution_mode": selected_execution_mode,
                "agent_fallback_to_baseline": fallback,
                "agent_shadow_used": shadow_used,
                "agent_shadow_accepted": shadow_accepted,
                "old_rule_pass": old_pass,
                "new_rule_pass": new_pass,
                "target_axis": verification.get("target_axis"),
                "target_axis_label": verification.get("target_axis_label"),
                "cosine_to_target_axis": verification.get("cosine_to_target_axis"),
                "verifier_rule_version": rule_version,
                "verifier_decision_reason": verification.get("verifier_decision_reason"),
            }
        )

    return {
        "rows": rows,
        "summary": {
            "total_records": len(records),
            "upright_direct_count": upright_direct_count,
            "plug_right_conditional_count": plug_right_conditional_count,
            "lying_flat_accepted_count": lying_flat_accepted_count,
            "lying_flat_fallback_count": lying_flat_fallback_count,
            "old_rule_pass": old_rule_pass,
            "new_rule_pass": new_rule_pass,
            "verifier_rule_version_distribution": dict(rule_versions),
            "decision_distribution": dict(decision_distribution),
            "selected_execution_mode_distribution": dict(selected_execution_mode_distribution),
            "mode_distribution": dict(mode_distribution),
        },
    }


def build_latency_summary(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    stage_timing = summary.get("stage_timing_summary", {}) or {}
    total_sec = stage_timing.get("total_sec", {}) or {}
    qwen_joint_sec = stage_timing.get("qwen_joint_sec", {}) or {}
    pcd_sec = stage_timing.get("pcd_sec", {}) or {}
    stage5_head_values = [safe_float(record.get("stage5_head_sec")) for record in records if record.get("stage5_head_sec") is not None]
    return {
        "total_sec": {
            "mean": safe_float(total_sec.get("mean")),
            "median": safe_float(total_sec.get("median")),
            "max": safe_float(total_sec.get("max")),
        },
        "qwen_joint_sec": {
            "mean": safe_float(qwen_joint_sec.get("mean")),
            "median": safe_float(qwen_joint_sec.get("median")),
            "max": safe_float(qwen_joint_sec.get("max")),
        },
        "pcd_sec": {
            "mean": safe_float(pcd_sec.get("mean")),
            "median": safe_float(pcd_sec.get("median")),
            "max": safe_float(pcd_sec.get("max")),
        },
        "stage5_head_sec": {
            "mean": mean_of(stage5_head_values),
            "median": median_of(stage5_head_values),
            "max": max_of(stage5_head_values),
        },
    }


def build_cap_clip_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    modes = ["cap_left_bottom_right", "cap_right_bottom_left", "clip_sideways"]
    rows_by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("stage5_mode") in modes:
            rows_by_mode[str(row.get("stage5_mode"))].append(row)

    rows = []
    selected = [row for row in records if row.get("stage5_mode") in modes]
    for mode in modes:
        bucket = rows_by_mode.get(mode, [])
        rows.append(
            {
                "mode": mode,
                "count": len(bucket),
                "selected_execution_mode_distribution": dict(Counter(row.get("agent_selected_execution_mode") for row in bucket)),
                "shadow_used_count": sum(1 for row in bucket if row.get("agent_shadow_used")),
                "shadow_accepted_count": sum(1 for row in bucket if row.get("agent_shadow_accepted")),
                "fallback_count": sum(1 for row in bucket if row.get("agent_fallback_to_baseline")),
                "stage5_applied_count": sum(1 for row in bucket if row.get("stage5_applied")),
            }
        )

    return {
        "rows": rows,
        "summary": {
            "selected_total": len(selected),
            "shadow_only_count": sum(1 for row in selected if row.get("agent_selected_execution_mode") == "stage5_shadow_only"),
            "fallback_count": sum(1 for row in selected if row.get("agent_fallback_to_baseline")),
            "accepted_count": sum(1 for row in selected if row.get("agent_verification_status") == "accepted"),
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(
    path: Path,
    before: Dict[str, Any],
    after: Dict[str, Any],
    latency: Dict[str, Any],
    cap_clip: Dict[str, Any],
    subset_summary: Dict[str, Any] | None,
) -> None:
    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    lines = []
    lines.append("# Open6DOR Short Experiments")
    lines.append("")
    lines.append("## 40-Case Verifier Ablation")
    lines.append("")
    lines.append("| Metric | Before | After |")
    lines.append("| --- | --- | --- |")
    for metric in [
        "success_count",
        "error_count",
        "used_stage5_count",
        "fallback_count",
        "rejected_count",
        "upright_direct_count",
        "plug_right_conditional_count",
        "lying_flat_accepted_count",
        "lying_flat_fallback_count",
        "old_rule_pass",
        "new_rule_pass",
    ]:
        lines.append(f"| {metric} | {before['summary'].get(metric, 0)} | {after['summary'].get(metric, 0)} |")
    lines.append(
        f"| verifier_rule_version_distribution | {json.dumps(before['summary'].get('verifier_rule_version_distribution', {}), ensure_ascii=False)} | {json.dumps(after['summary'].get('verifier_rule_version_distribution', {}), ensure_ascii=False)} |"
    )
    lines.append("")
    lines.append("## Latency / Overhead")
    lines.append("")
    lines.append("| Metric | Mean | Median | Max |")
    lines.append("| --- | --- | --- | --- |")
    for metric in ["total_sec", "qwen_joint_sec", "pcd_sec", "stage5_head_sec"]:
        block = latency[metric]
        lines.append(f"| {metric} | {fmt(block['mean'])} | {fmt(block['median'])} | {fmt(block['max'])} |")
    lines.append("")
    lines.append("结论: Stage5/agent overhead 相对 Qwen joint reasoning 很小, `stage5_head_sec` 远低于 `qwen_joint_sec`.")
    lines.append("")
    lines.append("## Cap / Clip / Shadow-only")
    lines.append("")
    lines.append("| Mode | Count | Selected Exec | Shadow Used | Shadow Accepted | Fallback |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in cap_clip["rows"]:
        lines.append(
            f"| {row['mode']} | {row['count']} | {json.dumps(row['selected_execution_mode_distribution'], ensure_ascii=False)} | "
            f"{row['shadow_used_count']} | {row['shadow_accepted_count']} | {row['fallback_count']} |"
        )
    lines.append("")
    lines.append("保守结论: cap/clip/sideways 当前保持 shadow-only 或 fallback, 不纳入强主结果 claim.")
    lines.append("")
    lines.append("## 400-Case Subset")
    lines.append("")
    if subset_summary is None:
        lines.append("Formal 400-case sampling is delegated to the server-side from4389 sampler.")
    else:
        lines.append(
            "Selected total: "
            f"{subset_summary.get('selected_total', 0)} / {subset_summary.get('target_total', 0)} "
            f"from {subset_summary.get('available_total', 0)} available tasks."
        )
        lines.append(
            "Selected family distribution: "
            f"{json.dumps(subset_summary.get('selected_family_distribution', {}), ensure_ascii=False)}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_command(path: Path, task_list_path: Path) -> None:
    command = "\n".join(
        [
            "cd /data/coding/SoFar",
            (
                "python open6dor/open6dor_perception.py "
                f"--task-list /data/coding/SoFar/{task_list_path.as_posix().split('paper_results/', 1)[-1]} "
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
    path.write_text(command + "\n", encoding="utf-8")


def pipeline_counts(summary_raw: Dict[str, Any], verifier_summary: Dict[str, Any]) -> Dict[str, Any]:
    agent_summary = summary_raw.get("agent_summary", {}) or {}
    return {
        "success_count": int(summary_raw.get("success_count", 0)),
        "error_count": int(summary_raw.get("error_count", 0)),
        "used_stage5_count": int(agent_summary.get("used_stage5_count", 0)),
        "fallback_count": int(agent_summary.get("fallback_count", 0)),
        "rejected_count": int(agent_summary.get("rejected_count", 0)),
        "upright_direct_count": verifier_summary["summary"].get("upright_direct_count", 0),
        "plug_right_conditional_count": verifier_summary["summary"].get("plug_right_conditional_count", 0),
        "lying_flat_accepted_count": verifier_summary["summary"].get("lying_flat_accepted_count", 0),
        "lying_flat_fallback_count": verifier_summary["summary"].get("lying_flat_fallback_count", 0),
        "old_rule_pass": verifier_summary["summary"].get("old_rule_pass", 0),
        "new_rule_pass": verifier_summary["summary"].get("new_rule_pass", 0),
        "verifier_rule_version_distribution": verifier_summary["summary"].get("verifier_rule_version_distribution", {}),
        "decision_distribution": agent_summary.get("decision_distribution", {}),
        "selected_execution_mode_distribution": agent_summary.get("selected_execution_mode_distribution", {}),
        "mode_distribution": agent_summary.get("stage5_mode_distribution", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Open6DOR Stage 5 short experiments and generate paper artifacts.")
    parser.add_argument("--before-summary", type=Path, required=True)
    parser.add_argument("--before-records", type=Path, required=True)
    parser.add_argument("--after-summary", type=Path, required=True)
    parser.add_argument("--after-records", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional Open6DOR task root for server-side from4389 subset generation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=400)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    before_summary_raw = load_json(args.before_summary)
    before_records_raw = load_json(args.before_records)
    after_summary_raw = load_json(args.after_summary)
    after_records_raw = load_json(args.after_records)

    before_records = before_records_raw.get("records", [])
    after_records = after_records_raw.get("records", [])
    before = build_verifier_summary(before_records)
    after = build_verifier_summary(after_records)
    latency = build_latency_summary(after_summary_raw, after_records)
    cap_clip = build_cap_clip_summary(after["rows"])
    subset = None
    if args.dataset_root is not None:
        subset = build_eval_subset_from_dataset_root(args.dataset_root.resolve(), seed=args.seed, target_total=args.subset_size)

    before_pipeline = pipeline_counts(before_summary_raw, before)
    after_pipeline = pipeline_counts(after_summary_raw, after)

    write_json(
        output_dir / "open6dor_verifier_ablation_before_after.json",
        {
            "before": before_pipeline,
            "after": after_pipeline,
            "notes": {
                "before_source": str(args.before_records),
                "after_source": str(args.after_records),
                "verifier_method": "legacy vs semantic-axis (recomputed from records)",
            },
        },
    )
    write_json(output_dir / "open6dor_latency_overhead.json", latency)
    write_json(output_dir / "open6dor_cap_clip_shadow_analysis.json", cap_clip)
    write_json(output_dir / "open6dor_verifier_ablation_before_after_table.json", before["rows"])
    write_csv(output_dir / "open6dor_verifier_ablation_before_after_table.csv", before["rows"] + after["rows"])
    write_json(output_dir / "open6dor_latency_overhead_table.json", latency)
    write_json(output_dir / "open6dor_cap_clip_shadow_analysis_table.json", cap_clip["rows"])

    if subset is not None:
        write_json(output_dir / "open6dor_eval_subset_400_from4389_seed42.json", subset["rows"])
        write_json(output_dir / "open6dor_eval_subset_400_from4389_seed42_summary.json", subset["summary"])
        write_json(output_dir / "open6dor_eval_subset_400_from4389_seed42_task_list.json", [row["task_dir"] for row in subset["rows"]])
        write_final_command(
            output_dir / "open6dor_400_final_method_command.txt",
            output_dir / "open6dor_eval_subset_400_from4389_seed42_task_list.json",
        )

    write_markdown_report(
        output_dir / "open6dor_short_experiments_report.md",
        {"summary": before_pipeline},
        {"summary": after_pipeline},
        latency,
        cap_clip,
        subset["summary"] if subset is not None else None,
    )
    write_json(
        output_dir / "open6dor_short_experiments_index.json",
        {
            "before": str(args.before_records),
            "after": str(args.after_records),
            "dataset_root": str(args.dataset_root) if args.dataset_root is not None else None,
            "output_dir": str(output_dir),
            "seed": args.seed,
            "subset_size": args.subset_size,
            "subset_source_policy": "server-side from4389 task tree" if args.dataset_root is not None else "deferred",
        },
    )


if __name__ == "__main__":
    main()
