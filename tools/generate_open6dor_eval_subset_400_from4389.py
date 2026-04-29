from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sofar.open6dor.eval_subset_sampling import build_eval_subset_from_dataset_root, write_json
from sofar.serve import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the Open6DOR 400-case mode-balanced subset from the full 4389 task tree.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=runtime_paths.open6dor_task_refine_6dof_dir(),
        help="Canonical Open6DOR task root to enumerate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=runtime_paths.resolve_from_root("paper_results", "open6dor_short_experiments_20260429"),
        help="Directory for the subset manifest, task list, and summary.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=400)
    return parser.parse_args()


def write_final_command(output_dir: Path, task_list_path: Path) -> None:
    command = "\n".join(
        [
            "cd /data/coding/SoFar",
            (
                "python open6dor/open6dor_perception.py "
                f"--task-list {task_list_path.as_posix()} "
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
    (output_dir / "open6dor_400_final_method_command.txt").write_text(command + "\n", encoding="utf-8")


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Open6DOR task root not found: {dataset_root}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = build_eval_subset_from_dataset_root(dataset_root, seed=args.seed, target_total=args.subset_size)
    rows = subset["rows"]
    summary = subset["summary"]

    manifest_path = output_dir / "open6dor_eval_subset_400_from4389_seed42.json"
    task_list_path = output_dir / "open6dor_eval_subset_400_from4389_seed42_task_list.json"
    summary_path = output_dir / "open6dor_eval_subset_400_from4389_seed42_summary.json"

    write_json(manifest_path, rows)
    write_json(task_list_path, [row["task_dir"] for row in rows])
    write_json(summary_path, summary)
    write_final_command(output_dir, task_list_path)

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
