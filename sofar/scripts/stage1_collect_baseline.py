import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect available Stage 1 baseline outputs into a unified summary."
    )
    parser.add_argument(
        "--hard-case-limit",
        type=int,
        default=100,
        help="Maximum number of hard cases to export.",
    )
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    fieldnames = [
        "source",
        "case_id",
        "track",
        "question_type",
        "status",
        "failure_type",
        "error_group",
        "error",
        "details",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_with_fields(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_error_message(message):
    if not message:
        return ""
    text = str(message).strip()
    replacements = [
        ("line 1 column 1 (char 0)", ""),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    text = " ".join(text.split())
    return text.strip(" :")


def timestamp_from_name(path):
    match = re.search(r"(\d{8}_\d{6})", path.name)
    return match.group(1) if match else ""


def extract_choice_letter(line):
    match = re.match(r"^\s*([ABCD])(?:[.\s]|$)", line.strip())
    if not match:
        return None
    return match.group(1)


def reconstruct_spatialbench_predictions():
    dataset_path = ROOT_DIR / "datasets" / "6dof_spatialbench" / "spatial_data.json"
    eval_path = OUTPUT_DIR / "eval_spatialbench.json"
    if not dataset_path.exists() or not eval_path.exists():
        return {
            "available": False,
            "incorrect_cases": [],
            "missing_prediction_ids": [],
        }

    dataset = load_json(dataset_path) or []
    eval_data = load_json(eval_path) or {}
    sample_results = {
        sample["id"]: sample
        for sample in eval_data.get("samples", [])
    }
    logs = sorted(OUTPUT_DIR.glob("eval_spatialbench_*.log"), key=timestamp_from_name)
    predictions_by_id = {}

    for log_path in logs:
        start_index = 0
        next_index = None
        current_index = None
        current_prediction = None

        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                stripped = line.strip()

                recovered_match = re.search(r"recovered (\d+) completed samples", stripped)
                if recovered_match:
                    start_index = int(recovered_match.group(1))
                    next_index = start_index
                    continue

                resume_match = re.search(r"resuming from progress file with (\d+) completed samples", stripped)
                if resume_match:
                    start_index = int(resume_match.group(1))
                    next_index = start_index
                    continue

                if "info: [" in line:
                    if next_index is None:
                        next_index = start_index
                    if current_index is not None and current_prediction is not None:
                        predictions_by_id[current_index] = {
                            "predicted_letter": current_prediction,
                            "source_log": log_path.name,
                        }
                    current_index = next_index
                    next_index += 1
                    current_prediction = None
                    continue

                failure_match = re.search(r"sample (\d+) failed:", stripped)
                if failure_match:
                    failed_id = int(failure_match.group(1))
                    predictions_by_id[failed_id] = {
                        "predicted_letter": None,
                        "source_log": log_path.name,
                    }
                    if current_index == failed_id:
                        current_index = None
                        current_prediction = None
                    continue

                if current_index is None:
                    continue

                letter = extract_choice_letter(stripped)
                if letter:
                    current_prediction = letter

        if current_index is not None and current_prediction is not None:
            predictions_by_id[current_index] = {
                "predicted_letter": current_prediction,
                "source_log": log_path.name,
            }

    letters = "ABCD"
    incorrect_cases = []
    missing_prediction_ids = []

    for item in dataset:
        sample_id = item["id"]
        result = sample_results.get(sample_id)
        if not result:
            continue
        if result.get("error") is not None:
            continue
        if result.get("correct") is not False:
            continue

        predicted = predictions_by_id.get(sample_id, {})
        predicted_letter = predicted.get("predicted_letter")
        if not predicted_letter:
            missing_prediction_ids.append(sample_id)
            predicted_text = None
        else:
            predicted_index = letters.index(predicted_letter)
            predicted_text = item["options"][predicted_index]

        correct_letter = letters[item["answer"]]
        correct_text = item["options"][item["answer"]]

        incorrect_cases.append(
            {
                "id": sample_id,
                "task_type": item["task_type"],
                "question_type": item["question_type"],
                "question": item["question"],
                "options": {
                    "A": item["options"][0],
                    "B": item["options"][1],
                    "C": item["options"][2],
                    "D": item["options"][3],
                },
                "predicted_letter": predicted_letter,
                "predicted_text": predicted_text,
                "correct_letter": correct_letter,
                "correct_text": correct_text,
                "source_log": predicted.get("source_log"),
            }
        )

    return {
        "available": True,
        "incorrect_cases": incorrect_cases,
        "missing_prediction_ids": missing_prediction_ids,
    }


def collect_spatialbench():
    path = OUTPUT_DIR / "eval_spatialbench.json"
    data = load_json(path)
    if not data:
        return {
            "available": False,
            "path": str(path),
            "summary": None,
            "hard_cases": [],
            "error_rows": [],
        }

    samples = data.get("samples", [])
    summary = data.get("summary", {})
    hard_cases = []
    error_rows = []

    for sample in samples:
        case_id = sample.get("id")
        task_type = sample.get("task_type")
        question_type = sample.get("question_type")
        error = sample.get("error")
        correct = sample.get("correct")
        status = "error" if error else ("correct" if correct else "incorrect")

        if error or correct is False:
            hard_cases.append(
                {
                    "source": "spatialbench",
                    "case_id": case_id,
                    "track": task_type,
                    "question_type": question_type,
                    "status": status,
                    "failure_type": "runtime_error" if error else "incorrect_answer",
                    "error_group": normalize_error_message(error) if error else "",
                    "error": error,
                    "note": "" if error else "Model completed inference but the answer was judged incorrect.",
                }
            )

        if error:
            error_rows.append(
                {
                    "source": "spatialbench",
                    "case_id": case_id,
                    "track": task_type,
                    "question_type": question_type,
                    "status": "error",
                    "failure_type": "runtime_error",
                    "error_group": normalize_error_message(error),
                    "error": error,
                    "details": "",
                }
            )

    return {
        "available": True,
        "path": str(path),
        "summary": summary,
        "hard_cases": hard_cases,
        "error_rows": error_rows,
    }


def summarize_open6dor_elapsed(records):
    success_values = [
        record.get("elapsed_sec")
        for record in records
        if record.get("status") == "success" and record.get("elapsed_sec") is not None
    ]
    if not success_values:
        return {
            "avg_elapsed_sec": None,
            "median_elapsed_sec": None,
            "min_elapsed_sec": None,
            "max_elapsed_sec": None,
        }
    return {
        "avg_elapsed_sec": round(sum(success_values) / len(success_values), 2),
        "median_elapsed_sec": round(statistics.median(success_values), 2),
        "min_elapsed_sec": round(min(success_values), 2),
        "max_elapsed_sec": round(max(success_values), 2),
    }


def collect_open6dor_run(source_name, progress_name, include_eval=False):
    progress_path = OUTPUT_DIR / progress_name
    progress = load_json(progress_path) or {}
    eval_pos = load_json(OUTPUT_DIR / "eval_pos.json") or {}
    eval_rot = load_json(OUTPUT_DIR / "eval_rot.json") or {}
    eval_6dof = load_json(OUTPUT_DIR / "eval_6dof.json") or {}

    records = progress.get("records", [])
    hard_cases = []
    error_rows = []

    for record in records:
        if record.get("status") != "error":
            continue
        task_dir = record.get("task_dir", "")
        hard_cases.append(
            {
                "source": source_name,
                "case_id": task_dir,
                "track": infer_open6dor_track(task_dir),
                "question_type": "",
                "status": "error",
                "failure_type": "runtime_error",
                "error_group": normalize_error_message(record.get("error")),
                "error": record.get("error"),
                "note": "",
            }
        )
        error_rows.append(
            {
                "source": source_name,
                "case_id": task_dir,
                "track": infer_open6dor_track(task_dir),
                "question_type": "",
                "status": "error",
                "failure_type": "runtime_error",
                "error_group": normalize_error_message(record.get("error")),
                "error": record.get("error"),
                "details": f"elapsed_sec={record.get('elapsed_sec', '')}; failed_stage={record.get('failed_stage', '')}",
            }
        )

    elapsed_summary = summarize_open6dor_elapsed(records)
    summary = {
        "progress_available": progress_path.exists(),
        "progress_path": str(progress_path),
        "run_mode": progress.get("run_mode"),
        "pilot": progress.get("pilot"),
        "task_list": progress.get("task_list"),
        "speed_profile": progress.get("speed_profile"),
        "total_tasks": progress.get("total_tasks"),
        "success_count": progress.get("success_count"),
        "error_count": progress.get("error_count"),
        "skipped_count": progress.get("skipped_count"),
        "processed_count": progress.get("processed_count"),
        "remaining_count": progress.get("remaining_count"),
        "avg_elapsed_sec": progress.get("avg_success_sec", elapsed_summary["avg_elapsed_sec"]),
        "median_elapsed_sec": progress.get("median_success_sec", elapsed_summary["median_elapsed_sec"]),
        "min_elapsed_sec": progress.get("min_success_sec", elapsed_summary["min_elapsed_sec"]),
        "max_elapsed_sec": progress.get("max_success_sec", elapsed_summary["max_elapsed_sec"]),
        "stage_timing_summary": progress.get("stage_timing_summary"),
        "is_partial": bool(progress) and progress.get("remaining_count", 0) not in (None, 0),
    }
    if include_eval:
        summary.update(
            {
                "eval_pos_available": bool(eval_pos),
                "eval_rot_available": bool(eval_rot),
                "eval_6dof_available": bool(eval_6dof),
                "eval_pos": eval_pos if eval_pos else None,
                "eval_rot": eval_rot if eval_rot else None,
                "eval_6dof": eval_6dof if eval_6dof else None,
            }
        )

    return {
        "available": bool(progress) or (include_eval and (bool(eval_pos) or bool(eval_rot) or bool(eval_6dof))),
        "summary": summary,
        "hard_cases": hard_cases,
        "error_rows": error_rows,
    }


def collect_open6dor():
    return {
        "full": collect_open6dor_run("open6dor_full", "open6dor_perception_progress.json", include_eval=True),
        "pilot_10": collect_open6dor_run("open6dor_pilot_10", "open6dor_perception_progress_open6dor10.json"),
    }


def infer_open6dor_track(task_dir):
    normalized = str(task_dir).replace("\\", "/")
    if "/task_refine_pos/" in normalized:
        return "position"
    if "/task_refine_rot/" in normalized:
        return "rotation"
    if "/task_refine_6dof/" in normalized:
        return "6dof"
    return "unknown"


def collect_hard_cases(spatialbench, open6dor, limit):
    ranked = []
    ranked.extend(spatialbench["hard_cases"])
    ranked.extend(open6dor["full"]["hard_cases"])
    ranked.extend(open6dor["pilot_10"]["hard_cases"])

    def sort_key(item):
        source_rank = 0 if item["source"] == "spatialbench" else 1
        status_rank = 0 if item["status"] == "error" else 1
        return (source_rank, status_rank, str(item["case_id"]))

    ranked = sorted(ranked, key=sort_key)
    return ranked[:limit]


def collect_all_hard_cases(spatialbench, open6dor):
    ranked = []
    ranked.extend(spatialbench["hard_cases"])
    ranked.extend(open6dor["full"]["hard_cases"])
    ranked.extend(open6dor["pilot_10"]["hard_cases"])

    def sort_key(item):
        source_rank = 0 if item["source"] == "spatialbench" else 1
        status_rank = 0 if item["status"] == "error" else 1
        return (source_rank, status_rank, str(item["case_id"]))

    return sorted(ranked, key=sort_key)


def build_baseline_results(spatialbench, open6dor):
    return {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(ROOT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "spatialbench": spatialbench["summary"],
        "open6dor_full_progress": open6dor["full"]["summary"],
        "open6dor_pilot_10": open6dor["pilot_10"]["summary"],
        "notes": {
            "spatialbench_complete": bool(
                spatialbench["summary"]
                and spatialbench["summary"].get("processed_samples") == 223
                and spatialbench["summary"].get("remaining_samples") == 0
            ),
            "open6dor_full_complete": bool(
                open6dor["full"]["summary"]
                and open6dor["full"]["summary"].get("remaining_count") == 0
                and open6dor["full"]["summary"].get("processed_count") not in (None, 0)
            ),
            "open6dor_eval_ready": bool(
                open6dor["full"]["summary"]
                and open6dor["full"]["summary"].get("eval_pos_available")
                and open6dor["full"]["summary"].get("eval_rot_available")
                and open6dor["full"]["summary"].get("eval_6dof_available")
            ),
            "open6dor_pilot_ready": bool(
                open6dor["pilot_10"]["summary"]
                and open6dor["pilot_10"]["summary"].get("processed_count") not in (None, 0)
            ),
            "stage1_mode": "baseline feasibility / pilot",
        },
    }


def build_error_case_summary(hard_cases, error_rows):
    runtime_groups = {}
    grouped_rows = defaultdict(list)
    for row in error_rows:
        key = (
            row.get("source", ""),
            row.get("track", ""),
            row.get("question_type", ""),
            row.get("error_group", ""),
        )
        grouped_rows[key].append(row)

    for key, rows in sorted(grouped_rows.items()):
        source, track, question_type, error_group = key
        runtime_groups[f"{source}::{track}::{question_type}::{error_group}"] = {
            "source": source,
            "track": track,
            "question_type": question_type,
            "error_group": error_group,
            "count": len(rows),
            "examples": [
                {
                    "case_id": row.get("case_id"),
                    "error": row.get("error"),
                    "details": row.get("details", ""),
                }
                for row in rows[:5]
            ],
        }

    incorrect_counter = Counter()
    incorrect_examples = defaultdict(list)
    for case in hard_cases:
        if case.get("failure_type") != "incorrect_answer":
            continue
        key = (
            case.get("source", ""),
            case.get("track", ""),
            case.get("question_type", ""),
        )
        incorrect_counter[key] += 1
        if len(incorrect_examples[key]) < 5:
            incorrect_examples[key].append(case.get("case_id"))

    incorrect_groups = []
    for key, count in sorted(incorrect_counter.items()):
        source, track, question_type = key
        incorrect_groups.append(
            {
                "source": source,
                "track": track,
                "question_type": question_type,
                "count": count,
                "example_case_ids": incorrect_examples[key],
            }
        )

    return {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_error_count": len(error_rows),
        "incorrect_answer_count": sum(1 for case in hard_cases if case.get("failure_type") == "incorrect_answer"),
        "runtime_error_groups": list(runtime_groups.values()),
        "incorrect_answer_groups": incorrect_groups,
    }


def main():
    args = parse_args()
    spatialbench = collect_spatialbench()
    open6dor = collect_open6dor()

    baseline_results = build_baseline_results(spatialbench, open6dor)
    all_hard_cases = collect_all_hard_cases(spatialbench, open6dor)
    hard_cases = all_hard_cases[:args.hard_case_limit]
    error_rows = (
        spatialbench["error_rows"]
        + open6dor["full"]["error_rows"]
        + open6dor["pilot_10"]["error_rows"]
    )
    error_case_summary = build_error_case_summary(all_hard_cases, error_rows)
    spatialbench_incorrect = reconstruct_spatialbench_predictions()

    write_json(OUTPUT_DIR / "baseline_results.json", baseline_results)
    write_json(
        OUTPUT_DIR / "hard_cases.json",
        {
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "hard_case_limit": args.hard_case_limit,
            "count": len(hard_cases),
            "total_available": len(all_hard_cases),
            "cases": hard_cases,
        },
    )
    write_csv(OUTPUT_DIR / "baseline_errors.csv", error_rows)
    write_json(OUTPUT_DIR / "error_case_summary.json", error_case_summary)
    write_json(
        OUTPUT_DIR / "spatialbench_incorrect_cases.json",
        {
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(spatialbench_incorrect["incorrect_cases"]),
            "missing_prediction_ids": spatialbench_incorrect["missing_prediction_ids"],
            "cases": spatialbench_incorrect["incorrect_cases"],
        },
    )
    write_csv_with_fields(
        OUTPUT_DIR / "spatialbench_incorrect_cases.csv",
        [
            {
                "id": case["id"],
                "task_type": case["task_type"],
                "question_type": case["question_type"],
                "question": case["question"],
                "predicted_letter": case["predicted_letter"],
                "predicted_text": case["predicted_text"],
                "correct_letter": case["correct_letter"],
                "correct_text": case["correct_text"],
                "source_log": case["source_log"],
            }
            for case in spatialbench_incorrect["incorrect_cases"]
        ],
        [
            "id",
            "task_type",
            "question_type",
            "question",
            "predicted_letter",
            "predicted_text",
            "correct_letter",
            "correct_text",
            "source_log",
        ],
    )

    print(f"[stage1] wrote {OUTPUT_DIR / 'baseline_results.json'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'hard_cases.json'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'baseline_errors.csv'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'error_case_summary.json'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'spatialbench_incorrect_cases.json'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'spatialbench_incorrect_cases.csv'}")
    print(
        "[stage1] summary: "
        f"spatialbench_available={spatialbench['available']}, "
        f"open6dor_full_available={open6dor['full']['available']}, "
        f"open6dor_pilot_10_available={open6dor['pilot_10']['available']}, "
        f"hard_cases={len(hard_cases)}, "
        f"errors={len(error_rows)}, "
        f"spatialbench_incorrect_cases={len(spatialbench_incorrect['incorrect_cases'])}"
    )


if __name__ == "__main__":
    main()
