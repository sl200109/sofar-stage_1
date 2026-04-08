import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


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


def collect_open6dor():
    progress_path = OUTPUT_DIR / "open6dor_perception_progress.json"
    eval_pos_path = OUTPUT_DIR / "eval_pos.json"
    eval_rot_path = OUTPUT_DIR / "eval_rot.json"
    eval_6dof_path = OUTPUT_DIR / "eval_6dof.json"

    progress = load_json(progress_path) or {}
    eval_pos = load_json(eval_pos_path) or {}
    eval_rot = load_json(eval_rot_path) or {}
    eval_6dof = load_json(eval_6dof_path) or {}

    records = progress.get("records", [])
    hard_cases = []
    error_rows = []

    for record in records:
        if record.get("status") != "error":
            continue
        task_dir = record.get("task_dir", "")
        hard_cases.append(
            {
                "source": "open6dor",
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
                "source": "open6dor",
                "case_id": task_dir,
                "track": infer_open6dor_track(task_dir),
                "question_type": "",
                "status": "error",
                "failure_type": "runtime_error",
                "error_group": normalize_error_message(record.get("error")),
                "error": record.get("error"),
                "details": f"elapsed_sec={record.get('elapsed_sec', '')}",
            }
        )

    summary = {
        "progress_available": progress_path.exists(),
        "progress_path": str(progress_path),
        "total_tasks": progress.get("total_tasks"),
        "success_count": progress.get("success_count"),
        "error_count": progress.get("error_count"),
        "skipped_count": progress.get("skipped_count"),
        "processed_count": progress.get("processed_count"),
        "remaining_count": progress.get("remaining_count"),
        "eval_pos_available": bool(eval_pos),
        "eval_rot_available": bool(eval_rot),
        "eval_6dof_available": bool(eval_6dof),
        "eval_pos": eval_pos if eval_pos else None,
        "eval_rot": eval_rot if eval_rot else None,
        "eval_6dof": eval_6dof if eval_6dof else None,
        "is_partial": bool(progress) and progress.get("remaining_count", 0) not in (None, 0),
    }

    return {
        "available": bool(progress) or bool(eval_pos) or bool(eval_rot) or bool(eval_6dof),
        "summary": summary,
        "hard_cases": hard_cases,
        "error_rows": error_rows,
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
    ranked.extend(open6dor["hard_cases"])

    def sort_key(item):
        source_rank = 0 if item["source"] == "spatialbench" else 1
        status_rank = 0 if item["status"] == "error" else 1
        return (source_rank, status_rank, str(item["case_id"]))

    ranked = sorted(ranked, key=sort_key)
    return ranked[:limit]


def build_baseline_results(spatialbench, open6dor):
    return {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(ROOT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "spatialbench": spatialbench["summary"],
        "open6dor": open6dor["summary"],
        "notes": {
            "spatialbench_complete": bool(
                spatialbench["summary"]
                and spatialbench["summary"].get("processed_samples") == 223
                and spatialbench["summary"].get("remaining_samples") == 0
            ),
            "open6dor_complete": bool(
                open6dor["summary"]
                and open6dor["summary"].get("remaining_count") == 0
                and open6dor["summary"].get("processed_count") not in (None, 0)
            ),
            "open6dor_eval_ready": bool(
                open6dor["summary"]
                and open6dor["summary"].get("eval_pos_available")
                and open6dor["summary"].get("eval_rot_available")
                and open6dor["summary"].get("eval_6dof_available")
            ),
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
    hard_cases = collect_hard_cases(spatialbench, open6dor, args.hard_case_limit)
    error_rows = spatialbench["error_rows"] + open6dor["error_rows"]
    error_case_summary = build_error_case_summary(hard_cases, error_rows)

    write_json(OUTPUT_DIR / "baseline_results.json", baseline_results)
    write_json(
        OUTPUT_DIR / "hard_cases.json",
        {
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "hard_case_limit": args.hard_case_limit,
            "count": len(hard_cases),
            "cases": hard_cases,
        },
    )
    write_csv(OUTPUT_DIR / "baseline_errors.csv", error_rows)
    write_json(OUTPUT_DIR / "error_case_summary.json", error_case_summary)

    print(f"[stage1] wrote {OUTPUT_DIR / 'baseline_results.json'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'hard_cases.json'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'baseline_errors.csv'}")
    print(f"[stage1] wrote {OUTPUT_DIR / 'error_case_summary.json'}")
    print(
        "[stage1] summary: "
        f"spatialbench_available={spatialbench['available']}, "
        f"open6dor_available={open6dor['available']}, "
        f"hard_cases={len(hard_cases)}, "
        f"errors={len(error_rows)}"
    )


if __name__ == "__main__":
    main()
