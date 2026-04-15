import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serve.stage5_manifest import build_stage5_smoke_manifests


def _load_symbol(module_path: Path, module_name: str, symbol_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


Stage4PointCacheDataset = _load_symbol(
    REPO_ROOT / "orientation" / "datasets" / "Stage4PointCache.py",
    "stage4_point_cache_dataset_module",
    "Stage4PointCacheDataset",
)
stage5_collate_fn = _load_symbol(
    REPO_ROOT / "orientation" / "datasets" / "Stage4PointCache.py",
    "stage4_point_cache_dataset_module_collate",
    "stage5_collate_fn",
)
PartConditionedOrientationHead = _load_symbol(
    REPO_ROOT / "orientation" / "models" / "PartConditionedOrientationHead.py",
    "part_conditioned_orientation_head_module",
    "PartConditionedOrientationHead",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 5 smoke manifest builder and dry-run entry.")
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest-dir", type=str, default="")
    parser.add_argument("--prefer-dataset", type=str, default="auto", choices=["auto", "spatialbench", "open6dor"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_manifest(summaries, prefer_dataset: str) -> Path:
    candidates = []
    for name in ["spatialbench", "open6dor"]:
        summary = summaries.get(name, {})
        available = int(summary.get("available_entries", 0))
        manifest_path = summary.get("manifest_path")
        if available > 0 and manifest_path:
            candidates.append((name, available, Path(manifest_path)))

    if not candidates:
        raise RuntimeError("No local Stage 4 cache entries are available for Stage 5 dry-run.")

    if prefer_dataset != "auto":
        for name, _, path in candidates:
            if name == prefer_dataset:
                return path
        raise RuntimeError(f"Preferred dataset '{prefer_dataset}' has no available local manifest entries.")

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][2]


def build_dataset(manifest_path: Path, num_points: int, max_samples: int):
    config = SimpleNamespace(
        MANIFEST_PATH=str(manifest_path),
        POINTS=num_points,
        subset="all",
        MAX_SAMPLES=max_samples,
    )
    return Stage4PointCacheDataset(config)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main():
    args = parse_args()
    set_seed(args.seed)

    repo_root = Path(args.repo_root).resolve()
    manifest_dir = Path(args.manifest_dir).resolve() if args.manifest_dir else (repo_root / "output")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    summaries = build_stage5_smoke_manifests(repo_root=repo_root, output_dir=manifest_dir)
    if args.manifest_only:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return

    manifest_path = select_manifest(summaries, args.prefer_dataset)
    dataset = build_dataset(manifest_path, args.num_points, args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError(f"Manifest {manifest_path} contains no usable entries for dry-run.")

    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, max(1, len(dataset))),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage5_collate_fn,
    )

    device = resolve_device(args.device)
    model_config = SimpleNamespace(
        point_dim=6,
        prior_dim=8,
        hidden_dim=128,
        text_dim=64,
        vocab_size=4096,
        max_tokens=32,
    )
    model = PartConditionedOrientationHead(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    batch = next(iter(dataloader))
    points = batch["points"].to(device)
    priors = batch["prior_vector"].to(device)
    target_direction = batch["target_direction"].to(device)
    instructions = batch["instruction"]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred_direction = model(points, instructions, priors)
    loss = 1 - torch.nn.functional.cosine_similarity(pred_direction, target_direction, dim=-1).mean()
    loss.backward()
    optimizer.step()

    summary = {
        "manifest_path": str(manifest_path),
        "dataset_size": len(dataset),
        "batch_size": int(points.shape[0]),
        "num_points": int(points.shape[1]),
        "device": device,
        "loss": round(float(loss.item()), 6),
        "sample_keys": [item.get("sample_key") for item in batch["meta"]],
        "summary_path": summaries.get("summary_path"),
    }

    summary_path = manifest_dir / "stage5_dry_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
