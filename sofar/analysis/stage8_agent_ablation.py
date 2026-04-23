import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["spatialbench", "open6dor", "both"], required=True)
    parser.add_argument("--agent-mode", choices=["off", "dataset", "auto"], default="dataset")
    parser.add_argument("--stage5-checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--reuse-source", action="store_true")
    return parser.parse_args()


def run_eval(dataset, mode, args):
    command = [
        sys.executable,
        "analysis/stage8_agent_eval.py",
        "--dataset",
        dataset,
        "--eval-mode",
        mode,
        "--agent-mode",
        args.agent_mode,
        "--output-dir",
        args.output_dir,
    ]
    if args.stage5_checkpoint:
        command += ["--stage5-checkpoint", args.stage5_checkpoint]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.reuse_existing:
        command += ["--reuse-existing"]
    if args.reuse_source:
        command += ["--reuse-source"]
    print("[stage8-ablation] running:", " ".join(command))
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def summary_file(output_dir: Path, dataset: str, mode: str) -> Path:
    return output_dir / dataset / mode / "summary.json"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    datasets = ["spatialbench", "open6dor"] if args.dataset == "both" else [args.dataset]
    modes = ["baseline", "direct_stage5", "agent"]

    ablation = {}
    for dataset in datasets:
        ablation[dataset] = {}
        for mode in modes:
            run_eval(dataset, mode, args)
            with summary_file(output_dir, dataset, mode).open("r", encoding="utf-8") as f:
                ablation[dataset][mode] = json.load(f)

    summary_path = output_dir / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "ablation_table.csv"
    fieldnames = [
        "dataset",
        "mode",
        "total_accuracy",
        "orientation_accuracy",
        "applicable_subset_accuracy",
        "valid_result_rate",
        "stage5_acceptance_rate",
        "fallback_rate",
        "shadow_acceptance_rate",
        "decision_distribution",
        "extra_latency",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, mode_map in ablation.items():
            for mode, summary in mode_map.items():
                writer.writerow(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "total_accuracy": summary.get("total_accuracy"),
                        "orientation_accuracy": summary.get("orientation_accuracy"),
                        "applicable_subset_accuracy": summary.get("applicable_subset_accuracy"),
                        "valid_result_rate": summary.get("valid_result_rate"),
                        "stage5_acceptance_rate": summary.get("stage5_acceptance_rate"),
                        "fallback_rate": summary.get("fallback_rate"),
                        "shadow_acceptance_rate": summary.get("shadow_acceptance_rate"),
                        "decision_distribution": json.dumps(
                            summary.get("decision_distribution", {}),
                            ensure_ascii=False,
                        ),
                        "extra_latency": summary.get("extra_latency"),
                    }
                )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "ablation_summary": str(summary_path),
                "ablation_table": str(csv_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
