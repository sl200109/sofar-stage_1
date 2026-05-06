import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SUPPORTED_METHODS = {
    "baseline_only",
    "pscr_rule_v2_safe",
    "pscr_rule_v3_verified",
    "pscr_shadow",
    "pscr_direct_no_verify",
}

SUMMARY_FIELDS = [
    "method",
    "run_id",
    "original_task_list",
    "effective_task_list",
    "max_tasks",
    "task_slice_mode",
    "method_dir",
    "perception_command",
    "eval_command",
    "method_notes",
    "total_task_count",
    "valid_result_count",
    "valid_result_rate",
    "position_l0",
    "position_l1",
    "rotation_l0",
    "rotation_l1",
    "rotation_l2",
    "six_dof_overall",
    "stage5_run_count",
    "stage5_used_count",
    "stage5_accepted_count",
    "stage5_rejected_count",
    "fallback_count",
    "reasoning_json_repaired_count",
    "reasoning_json_degraded_count",
    "avg_success_sec",
    "median_success_sec",
    "error_count",
    "eval_parse_status",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--task-list", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--speed-profile", type=str, default="conservative")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--stage5-checkpoint", type=str, default=None)
    parser.add_argument("--stage5-upright-expert-checkpoint", type=str, default=None)
    parser.add_argument("--stage5-flat-expert-checkpoint", type=str, default=None)
    parser.add_argument("--stage5-plug-expert-checkpoint", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-slice-mode", type=str, default="first")
    parser.add_argument("--task-slice-seed", type=int, default=42)
    return parser.parse_args()


def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_dataset_root(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    if dataset_root.name == "open6dor_v2":
        return dataset_root, dataset_root.parent
    nested = dataset_root / "open6dor_v2"
    if nested.exists():
        return nested.resolve(), dataset_root.resolve()
    return dataset_root, dataset_root.parent


def repo_relpath(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT_DIR)).replace("\\", "/")
    except Exception:
        return str(path)


def load_raw_task_list(task_list_path):
    with Path(task_list_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Task list must be a JSON list: {task_list_path}")
    for item in data:
        if not isinstance(item, str):
            raise ValueError(f"Task list entries must be strings: {item}")
    return data


def load_effective_task_entries(task_list_path):
    return load_raw_task_list(task_list_path)


def make_sliced_task_list(task_list_path, output_run_dir, max_tasks, mode="first", seed=42):
    if max_tasks is None:
        return Path(task_list_path), load_raw_task_list(task_list_path)
    if max_tasks <= 0:
        raise ValueError("--max-tasks must be positive")
    raw = load_raw_task_list(task_list_path)
    if mode != "first":
        raise ValueError(f"Unsupported task slice mode for now: {mode}")
    sliced = raw[:max_tasks]
    sliced_path = Path(output_run_dir) / f"task_list_first{max_tasks}.json"
    sliced_path.parent.mkdir(parents=True, exist_ok=True)
    with sliced_path.open("w", encoding="utf-8") as f:
        json.dump(sliced, f, indent=2, ensure_ascii=False)
    return sliced_path, sliced


def parse_methods(methods_arg):
    methods = [item.strip() for item in str(methods_arg or "").split(",") if item.strip()]
    if not methods:
        raise ValueError("No methods provided")
    unknown = [item for item in methods if item not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    return methods


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def detect_eval_script():
    candidates = [
        ROOT_DIR / "open6dor" / "eval_open6dor.py",
        ROOT_DIR.parent / "open6dor" / "eval_open6dor.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find eval_open6dor.py in sofar/open6dor or open6dor")


def detect_rule_v3_support():
    candidate = ROOT_DIR / "serve" / "semantic_orientation_agent.py"
    if not candidate.exists():
        return False
    text = candidate.read_text(encoding="utf-8")
    return "rule_v3_verified" in text


def detect_open6dor_agent_mode_support():
    candidate = ROOT_DIR / "open6dor" / "open6dor_perception.py"
    if not candidate.exists():
        return set()
    text = candidate.read_text(encoding="utf-8")
    match = re.search(
        r'--agent-mode".*?choices=\[([^\]]+)\]',
        text,
        flags=re.DOTALL,
    )
    if not match:
        return set()
    raw = match.group(1)
    modes = set(re.findall(r'"([^"]+)"|\'([^\']+)\'', raw))
    normalized = set()
    for left, right in modes:
        value = left or right
        if value:
            normalized.add(value)
    return normalized


def build_env_display(env):
    keys = ["SOFAR_DATASETS_DIR", "SOFAR_OUTPUT_DIR"]
    parts = []
    for key in keys:
        value = env.get(key)
        if value:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def command_to_string(command, env=None):
    env_prefix = build_env_display(env or {})
    cmd = " ".join(command)
    return f"{env_prefix} {cmd}".strip()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def build_method_command(method, args, effective_task_list_path):
    command = [
        args.python,
        repo_relpath(ROOT_DIR / "open6dor" / "open6dor_perception.py"),
        "--task-list",
        str(effective_task_list_path),
        "--reset-progress",
        "--rerun-existing",
        "--speed-profile",
        args.speed_profile,
    ]
    notes = []
    warnings = []
    runnable = True
    skip_reason = ""

    if method == "baseline_only":
        supported_agent_modes = detect_open6dor_agent_mode_support()
        if "off" in supported_agent_modes:
            command += ["--agent-mode", "off"]
            notes.append("baseline_only_agent_mode_off")
        elif "disabled" in supported_agent_modes:
            command += ["--agent-mode", "disabled"]
            notes.append("baseline_only_agent_mode_disabled")
        else:
            notes.append("baseline_only_no_stage5_no_agent_mode")
    elif method in {"pscr_rule_v2_safe", "pscr_shadow"}:
        command += [
            "--use-stage5-head",
            "--stage5-expert-routing",
            "task_family",
            "--agent-mode",
            "dataset",
            "--agent-policy",
            "rule_v2",
            "--agent-save-trace",
            "--agent-shadow-eval",
        ]
        if method == "pscr_shadow":
            notes.append("uses existing --agent-shadow-eval; final result behavior follows current agent policy")
    elif method == "pscr_rule_v3_verified":
        command += [
            "--use-stage5-head",
            "--stage5-expert-routing",
            "task_family",
            "--agent-mode",
            "dataset",
            "--agent-policy",
            "rule_v3_verified",
            "--agent-save-trace",
            "--agent-shadow-eval",
        ]
        warnings.append("pscr_rule_v3_verified requires rule_v3_verified implementation in semantic_orientation_agent.py")
        if not detect_rule_v3_support():
            runnable = False
            skip_reason = "skipped_missing_rule_v3"
    elif method == "pscr_direct_no_verify":
        warnings.append("pscr_direct_no_verify is reserved; current pipeline has no safe direct-no-verify switch")
        runnable = False
        skip_reason = "skipped_missing_direct_no_verify"
    else:
        raise ValueError(f"Unexpected method: {method}")

    checkpoint_args = [
        ("--stage5-checkpoint", args.stage5_checkpoint),
        ("--stage5-upright-expert-checkpoint", args.stage5_upright_expert_checkpoint),
        ("--stage5-flat-expert-checkpoint", args.stage5_flat_expert_checkpoint),
        ("--stage5-plug-expert-checkpoint", args.stage5_plug_expert_checkpoint),
    ]
    if "--use-stage5-head" in command:
        for flag, value in checkpoint_args:
            if value:
                command += [flag, value]

    return {
        "command": command,
        "notes": notes,
        "warnings": warnings,
        "runnable": runnable,
        "skip_reason": skip_reason,
    }


def load_task_dirs(task_entries, dataset_root):
    dirs = []
    for item in task_entries:
        path = Path(item)
        if not path.is_absolute():
            path = dataset_root / item
        dirs.append(path.resolve())
    return dirs


def task_relative_path(task_dir, dataset_root):
    return task_dir.resolve().relative_to(dataset_root.resolve())


def backup_original_results(task_dirs, dataset_root, backup_root):
    backup_root = ensure_dir(backup_root)
    manifest = {}
    for task_dir in task_dirs:
        rel = task_relative_path(task_dir, dataset_root)
        result_path = task_dir / "output" / "result.json"
        if result_path.exists():
            backup_path = backup_root / rel / "output" / "result.json"
            ensure_dir(backup_path.parent)
            shutil.copy2(result_path, backup_path)
            manifest[str(rel).replace("\\", "/")] = {"had_result": True}
        else:
            manifest[str(rel).replace("\\", "/")] = {"had_result": False}
    manifest_path = backup_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def restore_original_results(task_dirs, dataset_root, backup_root, manifest):
    backup_root = Path(backup_root)
    for task_dir in task_dirs:
        rel = str(task_relative_path(task_dir, dataset_root)).replace("\\", "/")
        result_path = task_dir / "output" / "result.json"
        backup_path = backup_root / Path(rel) / "output" / "result.json"
        had_result = bool((manifest.get(rel) or {}).get("had_result"))
        if had_result and backup_path.exists():
            ensure_dir(result_path.parent)
            shutil.copy2(backup_path, result_path)
        elif not had_result and result_path.exists():
            result_path.unlink()


def snapshot_method_results(method_dir, dataset_root, task_dirs):
    mirror_root = ensure_dir(Path(method_dir) / "eval_dataset_root" / "open6dor_v2")
    for task_dir in task_dirs:
        rel = task_relative_path(task_dir, dataset_root)
        source_config = task_dir / "task_config_new5.json"
        target_config = mirror_root / rel / "task_config_new5.json"
        ensure_dir(target_config.parent)
        shutil.copy2(source_config, target_config)
        source_result = task_dir / "output" / "result.json"
        target_result = mirror_root / rel / "output" / "result.json"
        if source_result.exists():
            ensure_dir(target_result.parent)
            shutil.copy2(source_result, target_result)
    return mirror_root


def run_command(command, cwd, env, stdout_path=None, stderr_path=None):
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path else None
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path else None
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=stdout_handle or subprocess.PIPE,
            stderr=stderr_handle or subprocess.PIPE,
        )
    finally:
        if stdout_handle:
            stdout_handle.close()
        if stderr_handle:
            stderr_handle.close()
    if stdout_path:
        stdout_text = stdout_path.read_text(encoding="utf-8")
    else:
        stdout_text = result.stdout or ""
    if stderr_path:
        stderr_text = stderr_path.read_text(encoding="utf-8")
    else:
        stderr_text = result.stderr or ""
    return result.returncode, stdout_text, stderr_text


def parse_float_from_match(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_eval_output(stdout_text, stderr_text, method_dir):
    metrics = {
        "position_l0": None,
        "position_l1": None,
        "rotation_l0": None,
        "rotation_l1": None,
        "rotation_l2": None,
        "six_dof_overall": None,
        "eval_parse_status": "failed",
    }
    combined = "\n".join([stdout_text or "", stderr_text or ""])
    explicit_patterns = {
        "position_l0": [r"Position L0\s*:\s*([0-9.]+)", r"position_l0\s*[:=]\s*([0-9.]+)", r"pos_l0\s*[:=]\s*([0-9.]+)"],
        "position_l1": [r"Position L1\s*:\s*([0-9.]+)", r"position_l1\s*[:=]\s*([0-9.]+)", r"pos_l1\s*[:=]\s*([0-9.]+)"],
        "rotation_l0": [r"Rotation L0\s*:\s*([0-9.]+)", r"rotation_l0\s*[:=]\s*([0-9.]+)", r"rot_l0\s*[:=]\s*([0-9.]+)"],
        "rotation_l1": [r"Rotation L1\s*:\s*([0-9.]+)", r"rotation_l1\s*[:=]\s*([0-9.]+)", r"rot_l1\s*[:=]\s*([0-9.]+)"],
        "rotation_l2": [r"Rotation L2\s*:\s*([0-9.]+)", r"rotation_l2\s*[:=]\s*([0-9.]+)", r"rot_l2\s*[:=]\s*([0-9.]+)"],
        "six_dof_overall": [r"6-DoF Overall\s*:\s*([0-9.]+)", r"six_dof_overall\s*[:=]\s*([0-9.]+)", r"6dof_overall\s*[:=]\s*([0-9.]+)", r"\boverall\s*[:=]\s*([0-9.]+)"],
    }
    for key, patterns in explicit_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, combined, flags=re.IGNORECASE)
            if match:
                metrics[key] = parse_float_from_match(match.group(1))
                break

    def section_text(start_marker, end_marker):
        start = combined.find(start_marker)
        if start < 0:
            return ""
        end = combined.find(end_marker, start)
        if end < 0:
            end = len(combined)
        return combined[start:end]

    if metrics["position_l0"] is None or metrics["position_l1"] is None:
        position_block = section_text("[eval_open6dor] evaluating position track", "[eval_open6dor] position track finished")
        if position_block:
            level1 = re.search(r"level1 acc:\s*([0-9.]+)", position_block, flags=re.IGNORECASE)
            level2 = re.search(r"level2 acc:\s*([0-9.]+)", position_block, flags=re.IGNORECASE)
            if metrics["position_l0"] is None and level1:
                metrics["position_l0"] = parse_float_from_match(level1.group(1))
            if metrics["position_l1"] is None and level2:
                metrics["position_l1"] = parse_float_from_match(level2.group(1))

    if metrics["rotation_l0"] is None or metrics["rotation_l1"] is None or metrics["rotation_l2"] is None:
        rotation_block = section_text("[eval_open6dor] evaluating rotation track", "[eval_open6dor] rotation track finished")
        if rotation_block:
            level1 = re.search(r"level1 acc:\s*([0-9.]+)", rotation_block, flags=re.IGNORECASE)
            level2 = re.search(r"level2 acc:\s*([0-9.]+)", rotation_block, flags=re.IGNORECASE)
            level3 = re.search(r"level3 acc:\s*([0-9.]+)", rotation_block, flags=re.IGNORECASE)
            if metrics["rotation_l0"] is None and level1:
                metrics["rotation_l0"] = parse_float_from_match(level1.group(1))
            if metrics["rotation_l1"] is None and level2:
                metrics["rotation_l1"] = parse_float_from_match(level2.group(1))
            if metrics["rotation_l2"] is None and level3:
                metrics["rotation_l2"] = parse_float_from_match(level3.group(1))

    if metrics["six_dof_overall"] is None:
        dof_block = section_text("[eval_open6dor] evaluating 6-dof track", "[eval_open6dor] 6-dof track finished")
        if dof_block:
            overall = re.search(r"6-dof all acc:\s*([0-9.]+)", dof_block, flags=re.IGNORECASE)
            if overall:
                metrics["six_dof_overall"] = parse_float_from_match(overall.group(1))

    found_count = sum(1 for key in SUMMARY_FIELDS if key in metrics and metrics[key] is not None)
    metric_found_count = sum(
        1 for key in ["position_l0", "position_l1", "rotation_l0", "rotation_l1", "rotation_l2", "six_dof_overall"]
        if metrics[key] is not None
    )
    if metric_found_count == 6:
        metrics["eval_parse_status"] = "complete"
    elif metric_found_count > 0:
        metrics["eval_parse_status"] = "partial"

    evaluator_output_dir = Path(method_dir) / "evaluator_output"
    if evaluator_output_dir.exists():
        for candidate in evaluator_output_dir.glob("*.json"):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                for key in ["position_l0", "position_l1", "rotation_l0", "rotation_l1", "rotation_l2", "six_dof_overall"]:
                    if metrics[key] is None and key in payload:
                        metrics[key] = parse_float_from_match(payload.get(key))
    return metrics


def is_valid_target_position(value):
    if not isinstance(value, list) or len(value) != 3:
        return False
    for item in value:
        if not isinstance(item, (int, float)):
            return False
    return True


def summarize_valid_results(method_dir, task_list):
    mirror_root = Path(method_dir) / "eval_dataset_root" / "open6dor_v2"
    total = len(task_list)
    valid = 0
    for item in task_list:
        rel = Path(item)
        if rel.is_absolute():
            parts = list(rel.parts)
            cut_idx = None
            for idx, part in enumerate(parts):
                if part in {"task_refine_pos", "task_refine_rot", "task_refine_6dof"}:
                    cut_idx = idx
                    break
            if cut_idx is None:
                continue
            rel = Path(*parts[cut_idx:])
        result_path = mirror_root / rel / "output" / "result.json"
        if not result_path.exists():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if is_valid_target_position(payload.get("target_position")):
            valid += 1
    return {
        "total_task_count": total,
        "valid_result_count": valid,
        "valid_result_rate": (valid / total) if total else None,
    }


def latest_matching_file(directory, pattern):
    matches = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def extract_pipeline_summary_fields(method_dir):
    result = {
        "stage5_run_count": None,
        "stage5_used_count": None,
        "stage5_accepted_count": None,
        "stage5_rejected_count": None,
        "fallback_count": None,
        "reasoning_json_repaired_count": None,
        "reasoning_json_degraded_count": None,
        "avg_success_sec": None,
        "median_success_sec": None,
        "error_count": None,
        "pipeline_records_payload": None,
    }
    method_dir = Path(method_dir)
    summary_path = latest_matching_file(method_dir, "open6dor_perception_summary*.json")
    records_path = latest_matching_file(method_dir, "stage5_open6dor_pipeline_records*.json")

    summary_payload = None
    records_payload = None
    if summary_path:
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_payload = None
    if records_path:
        try:
            records_payload = json.loads(records_path.read_text(encoding="utf-8"))
        except Exception:
            records_payload = None

    agent_summary = {}
    reasoning_json_summary = {}
    if isinstance(summary_payload, dict):
        agent_summary = summary_payload.get("agent_summary", {}) if isinstance(summary_payload.get("agent_summary"), dict) else {}
        reasoning_json_summary = summary_payload.get("reasoning_json_summary", {}) if isinstance(summary_payload.get("reasoning_json_summary"), dict) else {}
        result["avg_success_sec"] = summary_payload.get("avg_success_sec")
        result["median_success_sec"] = summary_payload.get("median_success_sec")
        result["error_count"] = summary_payload.get("error_count")
        result["stage5_run_count"] = first_not_none(
            agent_summary.get("total_records"),
            summary_payload.get("processed_count"),
            summary_payload.get("total_tasks"),
        )

    if isinstance(records_payload, dict):
        result["pipeline_records_payload"] = records_payload
        export_agent_summary = records_payload.get("agent_summary", {}) if isinstance(records_payload.get("agent_summary"), dict) else {}
        if export_agent_summary:
            agent_summary = export_agent_summary
        export_reasoning_summary = records_payload.get("reasoning_json_summary", {}) if isinstance(records_payload.get("reasoning_json_summary"), dict) else {}
        if export_reasoning_summary:
            reasoning_json_summary = export_reasoning_summary
        if result["stage5_run_count"] is None:
            result["stage5_run_count"] = records_payload.get("total_records")
        if result["error_count"] is None:
            result["error_count"] = records_payload.get("error_count")

    verification_distribution = agent_summary.get("verification_distribution", {}) if isinstance(agent_summary, dict) else {}
    result["stage5_used_count"] = first_not_none(
        agent_summary.get("used_stage5_count"),
        agent_summary.get("stage5_applied_count"),
    )
    result["stage5_accepted_count"] = first_not_none(verification_distribution.get("accepted"))
    result["stage5_rejected_count"] = first_not_none(
        agent_summary.get("rejected_count"),
        verification_distribution.get("rejected"),
    )
    result["fallback_count"] = agent_summary.get("fallback_count")
    result["reasoning_json_repaired_count"] = first_not_none(
        reasoning_json_summary.get("repaired_count"),
        agent_summary.get("reasoning_json_repaired_count"),
    )
    result["reasoning_json_degraded_count"] = first_not_none(
        reasoning_json_summary.get("degraded_count"),
        agent_summary.get("reasoning_json_degraded_count"),
    )
    return result


def classify_error_type(error_text):
    text = str(error_text or "").strip()
    if not text:
        return "unknown"
    first = text.split(":", 1)[0].strip()
    return first or text[:80]


def build_error_breakdown_rows(method, pipeline_records_payload):
    rows = []
    if not isinstance(pipeline_records_payload, dict):
        return rows
    for record in pipeline_records_payload.get("records", []) or []:
        if record.get("status") != "error":
            continue
        error_text = str(record.get("error") or "")
        rows.append(
            {
                "method": method,
                "failed_stage": str(record.get("failed_stage") or "unknown"),
                "error_type": classify_error_type(error_text),
                "example_error": error_text,
            }
        )
    grouped = {}
    for row in rows:
        key = (row["method"], row["failed_stage"], row["error_type"])
        if key not in grouped:
            grouped[key] = {
                "method": row["method"],
                "failed_stage": row["failed_stage"],
                "error_type": row["error_type"],
                "count": 0,
                "example_error": row["example_error"],
            }
        grouped[key]["count"] += 1
    return list(grouped.values())


def write_csv(path, rows, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_commands_file(path, command_records):
    lines = []
    for record in command_records:
        lines.append(f"[{record['method']}]")
        lines.append(f"original task_list = {record['original_task_list']}")
        lines.append(f"effective task_list = {record['effective_task_list']}")
        lines.append(f"max_tasks = {record['max_tasks']}")
        lines.append(f"perception = {record['perception_command']}")
        lines.append(f"eval = {record['eval_command']}")
        for warning in record.get("warnings", []):
            lines.append(f"warning = {warning}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_eval_command(args, eval_script_path):
    return [args.python, repo_relpath(eval_script_path)]


def run_method(method, args, dataset_root, datasets_dir, effective_task_list_path, effective_task_entries, output_run_dir, eval_script_path):
    method_dir = ensure_dir(Path(output_run_dir) / method)
    method_cfg = build_method_command(method, args, effective_task_list_path)
    perception_env = os.environ.copy()
    perception_env["SOFAR_DATASETS_DIR"] = str(datasets_dir)
    perception_env["SOFAR_OUTPUT_DIR"] = str(method_dir)
    perception_command_str = command_to_string(method_cfg["command"], env=perception_env)

    eval_command = build_eval_command(args, eval_script_path)
    mirror_parent = Path(method_dir) / "eval_dataset_root"
    eval_output_dir = Path(method_dir) / "evaluator_output"
    eval_env = os.environ.copy()
    eval_env["SOFAR_DATASETS_DIR"] = str(mirror_parent)
    eval_env["SOFAR_OUTPUT_DIR"] = str(eval_output_dir)
    eval_command_str = command_to_string(eval_command, env=eval_env)

    record = {
        "method": method,
        "run_id": Path(output_run_dir).name,
        "original_task_list": str(args.task_list),
        "effective_task_list": str(effective_task_list_path),
        "max_tasks": args.max_tasks,
        "task_slice_mode": args.task_slice_mode,
        "method_dir": str(method_dir),
        "perception_command": perception_command_str,
        "eval_command": eval_command_str,
        "method_notes": "; ".join(method_cfg["notes"] + method_cfg["warnings"]),
        "eval_parse_status": "failed",
    }

    if args.eval_only:
        pass
    elif args.skip_existing and latest_matching_file(method_dir, "open6dor_perception_summary*.json"):
        record["method_notes"] = (record["method_notes"] + "; " if record["method_notes"] else "") + "perception skipped due to --skip-existing"
    elif not method_cfg["runnable"]:
        record["eval_parse_status"] = method_cfg["skip_reason"] or "skipped"
    elif not args.dry_run:
        code, stdout_text, stderr_text = run_command(method_cfg["command"], cwd=ROOT_DIR, env=perception_env)
        (method_dir / "perception_stdout.txt").write_text(stdout_text, encoding="utf-8")
        (method_dir / "perception_stderr.txt").write_text(stderr_text, encoding="utf-8")
        if code != 0:
            record["method_notes"] = (record["method_notes"] + "; " if record["method_notes"] else "") + "perception command failed"
        snapshot_method_results(method_dir, dataset_root, load_task_dirs(effective_task_entries, dataset_root))

    if args.dry_run:
        if not method_cfg["runnable"]:
            record["eval_parse_status"] = method_cfg["skip_reason"] or "skipped"
        else:
            record["eval_parse_status"] = "dry_run"
        return record, []

    if not method_cfg["runnable"] and not args.eval_only:
        valid_summary = summarize_valid_results(method_dir, effective_task_entries)
        record.update(valid_summary)
        return record, []

    mirror_root = Path(method_dir) / "eval_dataset_root" / "open6dor_v2"
    if not mirror_root.exists():
        record["method_notes"] = (record["method_notes"] + "; " if record["method_notes"] else "") + "missing eval dataset mirror"
        valid_summary = summarize_valid_results(method_dir, effective_task_entries)
        record.update(valid_summary)
        record["eval_parse_status"] = "failed"
        return record, []

    ensure_dir(eval_output_dir)
    stdout_path = method_dir / "eval_stdout.txt"
    stderr_path = method_dir / "eval_stderr.txt"
    code, stdout_text, stderr_text = run_command(eval_command, cwd=ROOT_DIR, env=eval_env, stdout_path=stdout_path, stderr_path=stderr_path)
    eval_metrics = parse_eval_output(stdout_text, stderr_text, method_dir)
    if code != 0 and eval_metrics["eval_parse_status"] == "complete":
        eval_metrics["eval_parse_status"] = "partial"
    elif code != 0 and eval_metrics["eval_parse_status"] == "failed":
        eval_metrics["eval_parse_status"] = "failed"

    valid_summary = summarize_valid_results(method_dir, effective_task_entries)
    pipeline_summary = extract_pipeline_summary_fields(method_dir)
    record.update(valid_summary)
    record.update({k: eval_metrics.get(k) for k in ["position_l0", "position_l1", "rotation_l0", "rotation_l1", "rotation_l2", "six_dof_overall"]})
    record["eval_parse_status"] = eval_metrics.get("eval_parse_status", "failed")
    for key in [
        "stage5_run_count",
        "stage5_used_count",
        "stage5_accepted_count",
        "stage5_rejected_count",
        "fallback_count",
        "reasoning_json_repaired_count",
        "reasoning_json_degraded_count",
        "avg_success_sec",
        "median_success_sec",
        "error_count",
    ]:
        record[key] = pipeline_summary.get(key)
    error_rows = build_error_breakdown_rows(method, pipeline_summary.get("pipeline_records_payload"))
    return record, error_rows


def main():
    args = parse_args()
    dataset_root, datasets_dir = normalize_dataset_root(args.dataset_root)
    run_id = args.run_id or make_run_id()
    output_run_dir = ensure_dir(Path(args.output_root) / run_id)
    methods = parse_methods(args.methods)
    effective_task_list_path, effective_task_entries = make_sliced_task_list(
        args.task_list,
        output_run_dir,
        args.max_tasks,
        mode=args.task_slice_mode,
        seed=args.task_slice_seed,
    )
    eval_script_path = detect_eval_script()
    task_dirs = load_task_dirs(effective_task_entries, dataset_root)

    command_records = []
    summary_rows = []
    error_rows = []

    backup_manifest = None
    backup_root = output_run_dir / "_original_result_backup"
    should_backup = not args.dry_run and not args.eval_only
    if should_backup:
        backup_manifest = backup_original_results(task_dirs, dataset_root, backup_root)

    try:
        for method in methods:
            method_cfg = build_method_command(method, args, effective_task_list_path)
            command_records.append(
                {
                    "method": method,
                    "original_task_list": str(args.task_list),
                    "effective_task_list": str(effective_task_list_path),
                    "max_tasks": args.max_tasks,
                    "perception_command": command_to_string(
                        method_cfg["command"],
                        env={
                            "SOFAR_DATASETS_DIR": str(datasets_dir),
                            "SOFAR_OUTPUT_DIR": str(Path(output_run_dir) / method),
                        },
                    ),
                    "eval_command": command_to_string(
                        build_eval_command(args, eval_script_path),
                        env={
                            "SOFAR_DATASETS_DIR": str((Path(output_run_dir) / method / "eval_dataset_root")),
                            "SOFAR_OUTPUT_DIR": str(Path(output_run_dir) / method / "evaluator_output"),
                        },
                    ),
                    "warnings": method_cfg["warnings"],
                }
            )
            summary_row, method_error_rows = run_method(
                method,
                args,
                dataset_root,
                datasets_dir,
                effective_task_list_path,
                effective_task_entries,
                output_run_dir,
                eval_script_path,
            )
            for field in SUMMARY_FIELDS:
                summary_row.setdefault(field, None)
            summary_rows.append(summary_row)
            error_rows.extend(method_error_rows)
    finally:
        if should_backup and backup_manifest is not None:
            restore_original_results(task_dirs, dataset_root, backup_root, backup_manifest)

    summary_json_path = output_run_dir / "ablation_summary.json"
    summary_csv_path = output_run_dir / "ablation_summary.csv"
    error_csv_path = output_run_dir / "error_breakdown.csv"
    commands_path = output_run_dir / ("commands_dry_run.txt" if args.dry_run else "commands_executed.txt")

    summary_json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(summary_csv_path, summary_rows, SUMMARY_FIELDS)
    write_csv(error_csv_path, error_rows, ["method", "failed_stage", "error_type", "count", "example_error"])
    write_commands_file(commands_path, command_records)

    print(json.dumps(
        {
            "run_id": run_id,
            "original_task_list": str(args.task_list),
            "effective_task_list": str(effective_task_list_path),
            "max_tasks": args.max_tasks,
            "task_slice_mode": args.task_slice_mode,
            "output_run_dir": str(output_run_dir),
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "error_breakdown_csv": str(error_csv_path),
            "commands_file": str(commands_path),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
