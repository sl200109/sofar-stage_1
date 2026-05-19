import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath


ROOT_DIR = Path(__file__).resolve().parents[1]
OPEN6DOR_DIR = ROOT_DIR / "open6dor"


SPECIAL_DEVIATIONS = {"No annotation found", "Annotation stage 2"}
SIGN_FLIPS = {
    "x": (0,),
    "y": (1,),
    "z": (2,),
    "xy": (0, 1),
    "xyz": (0, 1, 2),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose Open6DOR baseline/PSCR oracle, gating, and axis-sign errors from existing results."
    )
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--task-list", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <run-root>/oracle_axis_analysis.",
    )
    return parser.parse_args()


def parse_methods(value):
    methods = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method")
    return methods


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def task_id_from_path(path):
    return PurePosixPath(str(path).replace("\\", "/")).name


def infer_mode_from_path(path):
    for part in PurePosixPath(str(path).replace("\\", "/")).parts:
        if ".__" in part:
            return part.rsplit(".__", 1)[-1]
    return "unknown"


def load_task_list(task_list_path):
    data = read_json(task_list_path)
    if not isinstance(data, list):
        raise ValueError(f"Task list must be a JSON list: {task_list_path}")
    return [str(item) for item in data]


def safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def normalize_eval_entry(entry):
    if not isinstance(entry, dict):
        raise ValueError(f"Per-task evaluator entry must be a dict, got: {type(entry).__name__}")
    pos_value = entry.get("pos_success", entry.get("success", 0))
    pos_pass = bool(int(pos_value)) if str(pos_value).strip() not in {"", "None"} else False
    deviation = entry.get("deviation")
    rot_pass = False
    if deviation not in SPECIAL_DEVIATIONS:
        deviation_float = safe_float(deviation)
        rot_pass = deviation_float is not None and int(deviation_float) <= 45
    return {
        **entry,
        "pos_pass": pos_pass,
        "rot_pass": rot_pass,
        "all_pass": bool(pos_pass and rot_pass),
    }


def load_per_task_eval(method_dir, output_dir=None):
    eval_path = Path(method_dir) / "evaluator_output" / "eval_6dof.json"
    if eval_path.exists():
        raw = read_json(eval_path)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict per-task eval in {eval_path}")
        return {str(task_id): normalize_eval_entry(entry) for task_id, entry in raw.items()}
    return recompute_per_task_eval(method_dir, output_dir=output_dir)


def load_method_results(method_dir):
    method_dir = Path(method_dir)
    result_files = sorted((method_dir / "eval_dataset_root").glob("open6dor_v2/**/output/result.json"))
    if not result_files:
        result_files = sorted(method_dir.glob("**/output/result.json"))
    results = {}
    for result_file in result_files:
        task_id = result_file.parent.parent.name
        if task_id in results:
            continue
        results[task_id] = {
            "result_path": str(result_file),
            "result": read_json(result_file),
        }
    return results


def load_config_paths(method_dir):
    method_dir = Path(method_dir)
    paths = sorted((method_dir / "eval_dataset_root").glob("open6dor_v2/task_refine_6dof/*/*/*/task_config_new5.json"))
    return {path.parent.name: path for path in paths}


def load_pipeline_records(method_dir):
    candidates = sorted(Path(method_dir).glob("stage5_open6dor_pipeline_records_*.json"))
    if not candidates:
        return {}
    # Prefer the stable non-timestamp artifact when both are present.
    candidates = sorted(candidates, key=lambda p: (len(p.name), p.name))
    payload = read_json(candidates[0])
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return {}
    by_task = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("task_dir"):
            continue
        task_id = task_id_from_path(record["task_dir"])
        by_task[task_id] = record
    return by_task


def parse_agent_signals(record):
    signals = (record or {}).get("agent_signals") or {}
    if isinstance(signals, str):
        try:
            signals = json.loads(signals)
        except json.JSONDecodeError:
            signals = {}
    return signals if isinstance(signals, dict) else {}


def infer_family_from_mode(mode):
    mode = str(mode or "").strip()
    if mode in {"lying_flat", "upside_down", "lower_rim"}:
        return "flat_upside_down_lying_flat"
    if mode in {"plug_right", "cap_left_bottom_right", "cap_right_bottom_left"}:
        return "plug_cap_sideways"
    if "upright" in mode:
        return "upright_vertical"
    if mode.endswith("_left") or mode.endswith("_right") or "_right_" in mode or "_left_" in mode:
        return "part_axis_left_right"
    return "unknown"


def build_task_metadata(task_entries, records_by_task):
    metadata = {}
    for entry in task_entries:
        task_id = task_id_from_path(entry)
        mode = infer_mode_from_path(entry)
        metadata[task_id] = {
            "task_id": task_id,
            "task_path": entry,
            "orientation_mode": mode,
            "task_family": infer_family_from_mode(mode),
            "stage5_semantic_status": "none",
            "stage5_used": False,
            "stage5_decision": "",
        }
    for task_id, record in records_by_task.items():
        signals = parse_agent_signals(record)
        mode = record.get("stage5_mode") or signals.get("orientation_mode") or metadata.get(task_id, {}).get("orientation_mode", "unknown")
        family = (
            record.get("stage5_checkpoint_family")
            or record.get("stage5_task_family")
            or signals.get("task_family")
            or infer_family_from_mode(mode)
        )
        metadata.setdefault(
            task_id,
            {
                "task_id": task_id,
                "task_path": record.get("task_dir", ""),
                "orientation_mode": mode,
                "task_family": family,
            },
        )
        metadata[task_id].update(
            {
                "orientation_mode": mode,
                "task_family": family,
                "stage5_semantic_status": record.get("stage5_semantic_status") or "none",
                "stage5_used": bool(record.get("agent_used_stage5")),
                "stage5_decision": record.get("agent_decision", ""),
            }
        )
    return metadata


def summarize_eval(eval_map, task_ids):
    task_ids = [task_id for task_id in task_ids if task_id in eval_map]
    total = len(task_ids)
    if total == 0:
        return {
            "n": 0,
            "pos_pass_count": 0,
            "rot_pass_count": 0,
            "all_pass_count": 0,
            "pos_acc": 0.0,
            "rot_acc": 0.0,
            "all_acc": 0.0,
        }
    pos_count = sum(1 for task_id in task_ids if eval_map[task_id]["pos_pass"])
    rot_count = sum(1 for task_id in task_ids if eval_map[task_id]["rot_pass"])
    all_count = sum(1 for task_id in task_ids if eval_map[task_id]["all_pass"])
    return {
        "n": total,
        "pos_pass_count": pos_count,
        "rot_pass_count": rot_count,
        "all_pass_count": all_count,
        "pos_acc": pos_count / total,
        "rot_acc": rot_count / total,
        "all_acc": all_count / total,
    }


def summarize_oracle(eval_maps, task_ids):
    task_ids = [task_id for task_id in task_ids if all(task_id in eval_map for eval_map in eval_maps.values())]
    total = len(task_ids)
    if total == 0:
        return summarize_eval({}, [])
    pos_count = sum(1 for task_id in task_ids if any(eval_map[task_id]["pos_pass"] for eval_map in eval_maps.values()))
    rot_count = sum(1 for task_id in task_ids if any(eval_map[task_id]["rot_pass"] for eval_map in eval_maps.values()))
    all_count = sum(1 for task_id in task_ids if any(eval_map[task_id]["all_pass"] for eval_map in eval_maps.values()))
    return {
        "n": total,
        "pos_pass_count": pos_count,
        "rot_pass_count": rot_count,
        "all_pass_count": all_count,
        "pos_acc": pos_count / total,
        "rot_acc": rot_count / total,
        "all_acc": all_count / total,
    }


def summarize_gated(baseline_eval, pscr_eval, task_ids, selected_ids):
    selected_ids = set(selected_ids)
    merged = {}
    for task_id in task_ids:
        merged[task_id] = pscr_eval[task_id] if task_id in selected_ids else baseline_eval[task_id]
    summary = summarize_eval(merged, task_ids)
    summary["selected_count"] = len([task_id for task_id in task_ids if task_id in selected_ids])
    summary["rot_wins_vs_baseline"] = sum(
        1
        for task_id in selected_ids
        if task_id in baseline_eval
        and task_id in pscr_eval
        and pscr_eval[task_id]["rot_pass"]
        and not baseline_eval[task_id]["rot_pass"]
    )
    summary["rot_losses_vs_baseline"] = sum(
        1
        for task_id in selected_ids
        if task_id in baseline_eval
        and task_id in pscr_eval
        and baseline_eval[task_id]["rot_pass"]
        and not pscr_eval[task_id]["rot_pass"]
    )
    summary["all_wins_vs_baseline"] = sum(
        1
        for task_id in selected_ids
        if task_id in baseline_eval
        and task_id in pscr_eval
        and pscr_eval[task_id]["all_pass"]
        and not baseline_eval[task_id]["all_pass"]
    )
    summary["all_losses_vs_baseline"] = sum(
        1
        for task_id in selected_ids
        if task_id in baseline_eval
        and task_id in pscr_eval
        and baseline_eval[task_id]["all_pass"]
        and not pscr_eval[task_id]["all_pass"]
    )
    return summary


def make_breakdown(group_name, task_ids, baseline_eval, pscr_eval, all_eval_maps, baseline_summary):
    task_ids = list(task_ids)
    baseline = summarize_eval(baseline_eval, task_ids)
    pscr = summarize_eval(pscr_eval, task_ids)
    oracle = summarize_oracle(all_eval_maps, task_ids)
    gated = summarize_gated(baseline_eval, pscr_eval, list(baseline_eval.keys()), task_ids)
    return {
        "group": group_name,
        "n": len(task_ids),
        "baseline_pos_acc": baseline["pos_acc"],
        "baseline_rot_acc": baseline["rot_acc"],
        "baseline_all_acc": baseline["all_acc"],
        "pscr_pos_acc": pscr["pos_acc"],
        "pscr_rot_acc": pscr["rot_acc"],
        "pscr_all_acc": pscr["all_acc"],
        "oracle_pos_acc": oracle["pos_acc"],
        "oracle_rot_acc": oracle["rot_acc"],
        "oracle_all_acc": oracle["all_acc"],
        "pscr_delta_rot": pscr["rot_acc"] - baseline["rot_acc"],
        "pscr_delta_all": pscr["all_acc"] - baseline["all_acc"],
        "gated_global_selected_count": gated["selected_count"],
        "gated_global_rot_acc": gated["rot_acc"],
        "gated_global_all_acc": gated["all_acc"],
        "gated_delta_rot_vs_baseline_global": gated["rot_acc"] - baseline_summary["rot_acc"],
        "gated_delta_all_vs_baseline_global": gated["all_acc"] - baseline_summary["all_acc"],
        "gated_rot_wins_vs_baseline": gated["rot_wins_vs_baseline"],
        "gated_rot_losses_vs_baseline": gated["rot_losses_vs_baseline"],
        "gated_all_wins_vs_baseline": gated["all_wins_vs_baseline"],
        "gated_all_losses_vs_baseline": gated["all_losses_vs_baseline"],
    }


def group_task_ids(metadata, key):
    grouped = defaultdict(list)
    for task_id, item in metadata.items():
        grouped[str(item.get(key) or "none")].append(task_id)
    return dict(grouped)


def load_default_evaluator():
    if str(OPEN6DOR_DIR) not in sys.path:
        sys.path.insert(0, str(OPEN6DOR_DIR))
    import evaluator  # type: ignore
    import numpy as np  # type: ignore
    from scipy.spatial.transform import Rotation as R  # type: ignore

    def rotation_evaluator(config_path, result_payload, task_id="", flip_name=""):
        pred_rotation = result_payload.get("transform_matrix") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        try:
            pred_quaternion = R.from_matrix(pred_rotation).as_quat()
        except Exception:
            pred_quaternion = np.array([1, 0, 0, 0])
        if np.isnan(pred_quaternion).any():
            pred_quaternion = np.array([1, 0, 0, 0])
        return evaluator.evaluate_rot(str(config_path), pred_quaternion)

    def position_evaluator(config_path, result_payload):
        task_config = read_json(config_path)
        init_obj_pos = [position[:3] for position in task_config["init_obj_pos"]]
        pos_tag = task_config["position_tag"]
        target_position = result_payload.get("target_position", [0, 0, 0])
        if pos_tag in ["left", "right", "front", "behind", "top"]:
            return evaluator.evaluate_posi(target_position, pos_tag, init_obj_pos[0])
        if pos_tag == "between":
            return evaluator.evaluate_posi(target_position, pos_tag, sel_pos_1=init_obj_pos[0], sel_pos_2=init_obj_pos[1])
        if pos_tag == "center":
            return evaluator.evaluate_posi(target_position, pos_tag, sel_pos_all=init_obj_pos[:-1])
        return evaluator.evaluate_posi(target_position, pos_tag, init_obj_pos[0])

    return rotation_evaluator, position_evaluator


def recompute_per_task_eval(method_dir, output_dir=None):
    method_dir = Path(method_dir)
    result_map = load_method_results(method_dir)
    config_paths = load_config_paths(method_dir)
    if not result_map or not config_paths:
        raise RuntimeError(
            f"No per-task evaluator output found for {method_dir}, and cannot recompute without "
            "eval_dataset_root/open6dor_v2 task configs and result.json files."
        )
    try:
        rotation_evaluator, position_evaluator = load_default_evaluator()
    except Exception as exc:
        raise RuntimeError(
            "No per-task evaluator output found, and importing Open6DOR evaluator failed. "
            "Run eval_open6dor.py first or sync evaluator dependencies."
        ) from exc

    eval_map = {}
    for task_id, result_item in result_map.items():
        if task_id not in config_paths:
            continue
        result_payload = result_item["result"]
        deviation = rotation_evaluator(config_paths[task_id], result_payload, task_id, "none")
        pos_pass = bool(position_evaluator(config_paths[task_id], result_payload))
        eval_map[task_id] = normalize_eval_entry(
            {
                "pos_success": int(pos_pass),
                "pred_position": result_payload.get("target_position"),
                "deviation": deviation,
            }
        )
    if output_dir:
        write_json(Path(output_dir) / f"{method_dir.name}_per_task_eval_export.json", eval_map)
    return eval_map


def normalize_vector(vector):
    import numpy as np  # type: ignore

    arr = np.array(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if not math.isfinite(float(norm)) or norm <= 1e-8:
        return arr
    return arr / norm


def single_vector_rotation(source, target):
    import numpy as np  # type: ignore

    a = normalize_vector(source)
    b = normalize_vector(target)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    norm = np.linalg.norm(cross)
    if norm <= 1e-8:
        if dot > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, a))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = normalize_vector(np.cross(a, axis))
        return rotation_from_axis_angle(axis, math.pi)
    skew = np.array(
        [
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0],
        ],
        dtype=float,
    )
    return np.eye(3) + skew + skew @ skew * ((1 - dot) / (norm**2))


def rotation_from_axis_angle(axis, angle):
    import numpy as np  # type: ignore

    axis = normalize_vector(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    c1 = 1 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def rotation_matrix_from_vectors(sources, targets):
    import numpy as np  # type: ignore

    if len(sources) == 1:
        return single_vector_rotation(sources[0], targets[0])
    a = np.array([normalize_vector(vector) for vector in sources], dtype=float)
    b = np.array([normalize_vector(vector) for vector in targets], dtype=float)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    matrix = vt.T @ u.T
    if np.linalg.det(matrix) < 0:
        vt[-1, :] *= -1
        matrix = vt.T @ u.T
    return matrix


def apply_orientation_sign_flip(result_payload, axes):
    import numpy as np  # type: ignore

    result_payload = copy.deepcopy(result_payload)
    axis_set = set(axes)
    target_orientation = result_payload.get("target_orientation") or {}
    if isinstance(target_orientation, dict) and target_orientation:
        flipped = {}
        for key, vector in target_orientation.items():
            values = list(vector)
            while len(values) < 3:
                values.append(0.0)
            for axis in axis_set:
                values[axis] = -float(values[axis])
            flipped[key] = values[:3]
        result_payload["target_orientation"] = flipped
        init_orientation = result_payload.get("init_orientation") or {}
        common = [key for key in flipped if key in init_orientation]
        if common:
            sources = [init_orientation[key] for key in common]
            targets = [flipped[key] for key in common]
            result_payload["transform_matrix"] = rotation_matrix_from_vectors(sources, targets).tolist()
            return result_payload

    matrix = np.array(result_payload.get("transform_matrix") or np.eye(3), dtype=float)
    diagonal = np.eye(3)
    for axis in axis_set:
        diagonal[axis, axis] = -1.0
    result_payload["transform_matrix"] = (diagonal @ matrix).tolist()
    return result_payload


def evaluate_sign_flips(pscr_results, pscr_eval, baseline_eval, config_paths, rotation_evaluator=None):
    if rotation_evaluator is None:
        rotation_evaluator, _ = load_default_evaluator()
    rows = []
    task_ids = sorted(task_id for task_id in pscr_eval if task_id in baseline_eval)
    for flip_name, axes in SIGN_FLIPS.items():
        flipped_eval = {}
        missing_config = []
        for task_id in task_ids:
            if task_id not in config_paths:
                missing_config.append(task_id)
                continue
            result_payload = pscr_results.get(task_id, {}).get("result")
            if result_payload is None:
                missing_config.append(task_id)
                continue
            flipped_payload = apply_orientation_sign_flip(result_payload, axes)
            deviation = rotation_evaluator(config_paths[task_id], flipped_payload, task_id, flip_name)
            flipped_eval[task_id] = normalize_eval_entry(
                {
                    "pos_success": int(pscr_eval[task_id]["pos_pass"]),
                    "pred_position": pscr_eval[task_id].get("pred_position"),
                    "deviation": deviation,
                }
            )
        if missing_config:
            raise RuntimeError(
                f"Cannot run sign-flip ablation {flip_name}: missing task configs/results for {len(missing_config)} tasks"
            )
        summary = summarize_eval(flipped_eval, task_ids)
        baseline_summary = summarize_eval(baseline_eval, task_ids)
        pscr_summary = summarize_eval(pscr_eval, task_ids)
        row = {
            "flip": flip_name,
            "n": summary["n"],
            "pos_acc": summary["pos_acc"],
            "rot_acc": summary["rot_acc"],
            "all_acc": summary["all_acc"],
            "rot_pass_count": summary["rot_pass_count"],
            "all_pass_count": summary["all_pass_count"],
            "delta_rot_vs_baseline": summary["rot_acc"] - baseline_summary["rot_acc"],
            "delta_all_vs_baseline": summary["all_acc"] - baseline_summary["all_acc"],
            "delta_rot_vs_pscr": summary["rot_acc"] - pscr_summary["rot_acc"],
            "delta_all_vs_pscr": summary["all_acc"] - pscr_summary["all_acc"],
            "rot_wins_vs_baseline": sum(
                1 for task_id in task_ids if flipped_eval[task_id]["rot_pass"] and not baseline_eval[task_id]["rot_pass"]
            ),
            "rot_losses_vs_baseline": sum(
                1 for task_id in task_ids if baseline_eval[task_id]["rot_pass"] and not flipped_eval[task_id]["rot_pass"]
            ),
            "all_wins_vs_baseline": sum(
                1 for task_id in task_ids if flipped_eval[task_id]["all_pass"] and not baseline_eval[task_id]["all_pass"]
            ),
            "all_losses_vs_baseline": sum(
                1 for task_id in task_ids if baseline_eval[task_id]["all_pass"] and not flipped_eval[task_id]["all_pass"]
            ),
        }
        rows.append(row)
    return rows


def build_breakdown_rows(grouped, baseline_eval, pscr_eval, eval_maps, baseline_summary):
    rows = []
    for group, task_ids in sorted(grouped.items()):
        rows.append(make_breakdown(group, task_ids, baseline_eval, pscr_eval, eval_maps, baseline_summary))
    return rows


def best_gating_candidates(rows, key="gated_delta_rot_vs_baseline_global", top_k=10):
    return sorted(rows, key=lambda row: (row.get(key, 0), row.get("gated_global_rot_acc", 0)), reverse=True)[:top_k]


def run_analysis(args, rotation_evaluator=None):
    run_root = Path(args.run_root)
    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "oracle_axis_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_entries = load_task_list(args.task_list)
    task_ids = [task_id_from_path(entry) for entry in task_entries]

    method_dirs = {method: run_root / method for method in methods}
    for method, method_dir in method_dirs.items():
        if not method_dir.exists():
            raise FileNotFoundError(f"Method directory does not exist: {method_dir}")

    eval_maps = {
        method: load_per_task_eval(method_dir, output_dir=output_dir)
        for method, method_dir in method_dirs.items()
    }
    result_maps = {method: load_method_results(method_dir) for method, method_dir in method_dirs.items()}

    baseline_method = "baseline_only" if "baseline_only" in methods else methods[0]
    pscr_method = next((method for method in methods if method != baseline_method), methods[-1])
    baseline_eval = eval_maps[baseline_method]
    pscr_eval = eval_maps[pscr_method]

    records_by_task = load_pipeline_records(method_dirs.get(pscr_method, run_root))
    metadata = build_task_metadata(task_entries, records_by_task)
    task_ids = [task_id for task_id in task_ids if task_id in baseline_eval and task_id in pscr_eval]

    method_metrics = {
        method: summarize_eval(eval_map, task_ids)
        for method, eval_map in eval_maps.items()
    }
    oracle = summarize_oracle(eval_maps, task_ids)
    baseline_summary = method_metrics[baseline_method]

    family_rows = build_breakdown_rows(
        group_task_ids(metadata, "task_family"),
        baseline_eval,
        pscr_eval,
        eval_maps,
        baseline_summary,
    )
    mode_rows = build_breakdown_rows(
        group_task_ids(metadata, "orientation_mode"),
        baseline_eval,
        pscr_eval,
        eval_maps,
        baseline_summary,
    )
    semantic_rows = build_breakdown_rows(
        group_task_ids(metadata, "stage5_semantic_status"),
        baseline_eval,
        pscr_eval,
        eval_maps,
        baseline_summary,
    )

    config_paths = load_config_paths(method_dirs[pscr_method])
    sign_flip_rows = evaluate_sign_flips(
        result_maps[pscr_method],
        pscr_eval,
        baseline_eval,
        config_paths,
        rotation_evaluator=rotation_evaluator,
    )

    summary_rows = []
    for method in methods:
        row = {"label": method, **method_metrics[method]}
        summary_rows.append(row)
    summary_rows.append({"label": "oracle_union", **oracle})

    summary = {
        "run_root": str(run_root),
        "task_list": str(args.task_list),
        "methods": methods,
        "baseline_method": baseline_method,
        "pscr_method": pscr_method,
        "method_metrics": method_metrics,
        "oracle_union": oracle,
        "best_family_gates_by_rot": best_gating_candidates(family_rows),
        "best_mode_gates_by_rot": best_gating_candidates(mode_rows),
        "best_semantic_status_gates_by_rot": best_gating_candidates(semantic_rows),
        "sign_flip_ablation": sign_flip_rows,
        "notes": [
            "All pass/fail values come from per-task evaluator output or evaluator recomputation.",
            "Oracle union is an upper bound: a task passes if any input method passes.",
            "Gated-global rows mean: use PSCR only for this group, otherwise keep baseline.",
            "This script does not know the original SoFar paper target metrics; compare these outputs externally.",
        ],
    }

    summary_fields = [
        "label",
        "n",
        "pos_pass_count",
        "rot_pass_count",
        "all_pass_count",
        "pos_acc",
        "rot_acc",
        "all_acc",
    ]
    breakdown_fields = [
        "group",
        "n",
        "baseline_pos_acc",
        "baseline_rot_acc",
        "baseline_all_acc",
        "pscr_pos_acc",
        "pscr_rot_acc",
        "pscr_all_acc",
        "oracle_pos_acc",
        "oracle_rot_acc",
        "oracle_all_acc",
        "pscr_delta_rot",
        "pscr_delta_all",
        "gated_global_selected_count",
        "gated_global_rot_acc",
        "gated_global_all_acc",
        "gated_delta_rot_vs_baseline_global",
        "gated_delta_all_vs_baseline_global",
        "gated_rot_wins_vs_baseline",
        "gated_rot_losses_vs_baseline",
        "gated_all_wins_vs_baseline",
        "gated_all_losses_vs_baseline",
    ]
    sign_flip_fields = [
        "flip",
        "n",
        "pos_acc",
        "rot_acc",
        "all_acc",
        "rot_pass_count",
        "all_pass_count",
        "delta_rot_vs_baseline",
        "delta_all_vs_baseline",
        "delta_rot_vs_pscr",
        "delta_all_vs_pscr",
        "rot_wins_vs_baseline",
        "rot_losses_vs_baseline",
        "all_wins_vs_baseline",
        "all_losses_vs_baseline",
    ]

    write_json(output_dir / "oracle_summary.json", summary)
    write_csv(output_dir / "oracle_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "family_breakdown.csv", family_rows, breakdown_fields)
    write_csv(output_dir / "mode_breakdown.csv", mode_rows, breakdown_fields)
    write_csv(output_dir / "semantic_status_breakdown.csv", semantic_rows, breakdown_fields)
    write_csv(output_dir / "sign_flip_ablation.csv", sign_flip_rows, sign_flip_fields)
    return summary


def main():
    summary = run_analysis(parse_args())
    print(json.dumps(summary["method_metrics"], indent=2, ensure_ascii=False))
    print(json.dumps({"oracle_union": summary["oracle_union"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
