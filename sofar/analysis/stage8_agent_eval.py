import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve.agent_debug import write_agent_eval_bundle
from serve.batch_logging import make_run_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["spatialbench", "open6dor", "both"], required=True)
    parser.add_argument("--eval-mode", choices=["baseline", "direct_stage5", "agent"], required=True)
    parser.add_argument("--agent-mode", choices=["off", "dataset", "auto"], default="dataset")
    parser.add_argument("--stage5-checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--reuse-source",
        action="store_true",
        help="Reuse the latest dataset source summary/progress file instead of rerunning the underlying pipeline.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return ROOT_DIR


def default_output_root() -> Path:
    return repo_root() / "output" / "agent_eval" / make_run_id()


def run_command(command):
    print("[stage8-eval] running:", " ".join(command))
    start = time.perf_counter()
    subprocess.run(command, cwd=repo_root(), check=True)
    return round(time.perf_counter() - start, 2)


def _bucket_accuracy(records):
    total = len(records)
    return {
        "count": total,
        "accuracy": (sum(1 for item in records if item.get("correct")) / total) if total else None,
        "used_stage5_count": sum(1 for item in records if item.get("agent_used_stage5")),
    }


def _spatialbench_bucket(sample):
    if sample.get("task_type") != "orientation":
        return "non_orientation"
    return sample.get("agent_stage5_category") or "unlabeled_orientation"


def build_spatialbench_eval(source, wall_time_sec, eval_mode, agent_mode):
    samples = source.get("samples", [])
    orientation_samples = [item for item in samples if item.get("task_type") == "orientation"]
    applicable_samples = [
        item for item in orientation_samples if item.get("agent_stage5_category") == "single_object_direction"
    ]

    per_sample_records = []
    buckets = {}
    for sample in samples:
        bucket = _spatialbench_bucket(sample)
        record = {
            "sample_key": str(sample.get("id")),
            "task_type": sample.get("task_type"),
            "question_type": sample.get("question_type"),
            "orientation_mode": "",
            "bucket": bucket,
            "correct": bool(sample.get("correct")),
            "agent_controller": sample.get("agent_controller"),
            "agent_selected_dataset_agent": sample.get("agent_selected_dataset_agent"),
            "agent_decision": sample.get("agent_decision"),
            "agent_decision_reason": sample.get("agent_decision_reason"),
            "agent_verification_status": sample.get("agent_verification_status"),
            "agent_selected_execution_mode": sample.get("agent_selected_execution_mode"),
            "agent_used_stage5": bool(sample.get("agent_used_stage5")),
            "agent_fallback_to_baseline": bool(sample.get("agent_fallback_to_baseline")),
            "agent_shadow_used": bool(sample.get("agent_shadow_used")),
            "agent_shadow_accepted": bool(sample.get("agent_shadow_accepted")),
        }
        per_sample_records.append(record)
        buckets.setdefault(bucket, []).append(record)

    metrics_by_bucket = {bucket: _bucket_accuracy(records) for bucket, records in buckets.items()}
    source_summary = source.get("summary", {})
    summary = {
        "dataset": "spatialbench",
        "eval_mode": eval_mode,
        "agent_mode": agent_mode,
        "processed_samples": len(samples),
        "run_wall_time_sec": wall_time_sec,
        "extra_latency": wall_time_sec / max(1, len(samples)),
        "total_accuracy": source_summary.get("total_accuracy"),
        "orientation_accuracy": (
            sum(1 for item in orientation_samples if item.get("correct")) / len(orientation_samples)
            if orientation_samples
            else None
        ),
        "applicable_subset_accuracy": (
            sum(1 for item in applicable_samples if item.get("correct")) / len(applicable_samples)
            if applicable_samples
            else None
        ),
        "decision_distribution": source_summary.get("agent_decision_distribution", {}),
        "stage5_usage_precision": source_summary.get("stage5_usage_precision"),
        "fallback_precision": source_summary.get("fallback_precision"),
        "reverification_trigger_rate": source_summary.get("agent_triggered_reverification_count", 0)
        / max(1, len(samples)),
    }
    return summary, per_sample_records, metrics_by_bucket


def _open6dor_bucket(record):
    mode = str(record.get("stage5_mode") or "").strip().lower()
    if mode in {"upright", "upright_lens_forth", "plug_right", "lying_flat"}:
        return mode
    return "others"


def _estimate_wall_time_from_source(source):
    records = source.get("records", []) or source.get("samples", []) or []
    total = 0.0
    found = False
    for item in records:
        for field in ("total_sec", "elapsed_sec"):
            value = item.get(field)
            if value is not None:
                try:
                    total += float(value)
                    found = True
                    break
                except Exception:
                    continue
    if found:
        return round(total, 2)

    processed = source.get("processed_count") or source.get("total_tasks") or len(records)
    avg_success = source.get("avg_success_sec")
    if avg_success is not None:
        try:
            return round(float(avg_success) * max(1, int(processed)), 2)
        except Exception:
            pass
    return 0.0


def _source_path_for_dataset(dataset):
    if dataset == "spatialbench":
        return repo_root() / "output" / "eval_spatialbench_progress.json"
    return repo_root() / "output" / "open6dor_perception_summary.json"


def build_open6dor_eval(source, wall_time_sec, eval_mode, agent_mode):
    records = source.get("records", [])
    per_sample_records = []
    buckets = {}
    for record in records:
        bucket = _open6dor_bucket(record)
        row = {
            "sample_key": record.get("sample_key") or record.get("task_dir"),
            "task_type": "manipulation",
            "question_type": "",
            "orientation_mode": record.get("stage5_mode"),
            "bucket": bucket,
            "correct": record.get("status") == "success",
            "agent_controller": record.get("agent_controller"),
            "agent_selected_dataset_agent": record.get("agent_selected_dataset_agent"),
            "agent_decision": record.get("agent_decision"),
            "agent_decision_reason": record.get("agent_decision_reason"),
            "agent_verification_status": record.get("agent_verification_status"),
            "agent_selected_execution_mode": record.get("agent_selected_execution_mode"),
            "agent_used_stage5": bool(record.get("agent_used_stage5")),
            "agent_fallback_to_baseline": bool(record.get("agent_fallback_to_baseline")),
            "agent_shadow_used": bool(record.get("agent_shadow_used")),
            "agent_shadow_accepted": bool(record.get("agent_shadow_accepted")),
        }
        per_sample_records.append(row)
        buckets.setdefault(bucket, []).append(row)

    metrics_by_bucket = {bucket: _bucket_accuracy(rows) for bucket, rows in buckets.items()}
    total = len(records)
    triggered = sum(1 for item in records if item.get("agent_triggered_reverification"))
    rejected = sum(1 for item in records if item.get("agent_verification_status") == "rejected")
    shadow_used = sum(1 for item in records if item.get("agent_shadow_used"))
    shadow_accepted = sum(1 for item in records if item.get("agent_shadow_accepted"))
    summary = {
        "dataset": "open6dor",
        "eval_mode": eval_mode,
        "agent_mode": agent_mode,
        "processed_samples": total,
        "run_wall_time_sec": wall_time_sec,
        "extra_latency": wall_time_sec / max(1, total),
        "valid_result_rate": sum(1 for item in records if item.get("status") == "success") / max(1, total),
        "stage5_acceptance_rate": sum(1 for item in records if item.get("agent_used_stage5")) / max(1, total),
        "shadow_acceptance_rate": shadow_accepted / max(1, shadow_used),
        "fallback_rate": sum(1 for item in records if item.get("agent_fallback_to_baseline")) / max(1, total),
        "corrected_error_count": sum(
            1
            for item in records
            if item.get("agent_decision") == "reject_stage5_keep_parser_orientation"
            and item.get("status") == "success"
        ),
        "over_trigger_rate": rejected / max(1, triggered),
        "decision_distribution": source.get("agent_summary", {}).get("decision_distribution", {}),
        "stage5_mode_distribution": source.get("agent_summary", {}).get("stage5_mode_distribution", {}),
    }
    return summary, per_sample_records, metrics_by_bucket


def spatialbench_command(args):
    command = [sys.executable, "spatialbench/eval_spatialbench.py", "--reset-progress", "--speed-profile", "conservative"]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.eval_mode in {"direct_stage5", "agent"}:
        command += ["--use-stage5-head"]
        if args.stage5_checkpoint:
            command += ["--stage5-checkpoint", args.stage5_checkpoint]
    if args.eval_mode == "direct_stage5":
        command += ["--agent-mode", "off"]
    elif args.eval_mode == "agent":
        command += ["--agent-mode", args.agent_mode, "--agent-save-trace"]
    return command


def open6dor_command(args):
    command = [
        sys.executable,
        "open6dor/open6dor_perception.py",
        "--reset-progress",
        "--rerun-existing",
        "--speed-profile",
        "conservative",
    ]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.eval_mode in {"direct_stage5", "agent"}:
        command += ["--use-stage5-head"]
        if args.stage5_checkpoint:
            command += ["--stage5-checkpoint", args.stage5_checkpoint]
    if args.eval_mode == "direct_stage5":
        command += ["--agent-mode", "off", "--stage5-open6dor-modes", "all"]
    elif args.eval_mode == "agent":
        command += ["--agent-mode", args.agent_mode, "--agent-save-trace", "--agent-shadow-eval"]
    return command


def evaluate_one(dataset, args, output_root):
    dataset_dir = output_root / dataset / args.eval_mode
    summary_path = dataset_dir / "summary.json"
    if args.reuse_existing and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    source_path = _source_path_for_dataset(dataset)
    if args.reuse_source:
        if not source_path.exists():
            raise FileNotFoundError(f"reuse-source requested but source file is missing: {source_path}")
        with source_path.open("r", encoding="utf-8") as f:
            source = json.load(f)
        wall_time = _estimate_wall_time_from_source(source)
        print(f"[stage8-eval] reusing source: {source_path}")
        if dataset == "spatialbench":
            summary, per_sample_records, metrics_by_bucket = build_spatialbench_eval(
                source, wall_time, args.eval_mode, args.agent_mode
            )
        else:
            summary, per_sample_records, metrics_by_bucket = build_open6dor_eval(
                source, wall_time, args.eval_mode, args.agent_mode
            )
        write_agent_eval_bundle(
            output_dir=output_root,
            dataset=dataset,
            mode=args.eval_mode,
            summary=summary,
            per_sample_records=per_sample_records,
            metrics_by_bucket=metrics_by_bucket,
        )
        return summary

    if dataset == "spatialbench":
        wall_time = run_command(spatialbench_command(args))
        with source_path.open("r", encoding="utf-8") as f:
            source = json.load(f)
        summary, per_sample_records, metrics_by_bucket = build_spatialbench_eval(
            source, wall_time, args.eval_mode, args.agent_mode
        )
    else:
        wall_time = run_command(open6dor_command(args))
        with source_path.open("r", encoding="utf-8") as f:
            source = json.load(f)
        summary, per_sample_records, metrics_by_bucket = build_open6dor_eval(
            source, wall_time, args.eval_mode, args.agent_mode
        )

    write_agent_eval_bundle(
        output_dir=output_root,
        dataset=dataset,
        mode=args.eval_mode,
        summary=summary,
        per_sample_records=per_sample_records,
        metrics_by_bucket=metrics_by_bucket,
    )
    return summary


def main():
    args = parse_args()
    output_root = Path(args.output_dir) if args.output_dir else default_output_root()
    datasets = ["spatialbench", "open6dor"] if args.dataset == "both" else [args.dataset]

    combined = {}
    for dataset in datasets:
        combined[dataset] = evaluate_one(dataset, args, output_root)

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(json.dumps({"output_dir": str(output_root), "datasets": combined}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
