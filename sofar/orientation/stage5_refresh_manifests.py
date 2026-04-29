import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serve.stage5_manifest import build_stage5_smoke_manifests


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild Stage 5 smoke manifests from current Stage 4 cache/records.")
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "output")
    summaries = build_stage5_smoke_manifests(repo_root=repo_root, output_dir=output_dir)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
