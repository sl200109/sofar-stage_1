import argparse
import importlib.util
import json
import sys
from pathlib import Path

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
    "stage4_point_cache_dataset_eval_module",
    "Stage4PointCacheDataset",
)
stage5_collate_fn = _load_symbol(
    REPO_ROOT / "orientation" / "datasets" / "Stage4PointCache.py",
    "stage4_point_cache_dataset_eval_collate_module",
    "stage5_collate_fn",
)
PartConditionedOrientationHead = _load_symbol(
    REPO_ROOT / "orientation" / "models" / "PartConditionedOrientationHead.py",
    "part_conditioned_orientation_head_eval_module",
    "PartConditionedOrientationHead",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 5 checkpoint on unseen samples.")
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--dataset-mode", choices=["combined", "spatialbench", "open6dor"], default="combined")
    parser.add_argument("--subset", choices=["val", "test"], default="test")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _load_jsonl(path: Path):
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_jsonl(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _choose_manifests(summary: dict, dataset_mode: str):
    chosen = []
    for name in ["spatialbench", "open6dor"]:
        if dataset_mode != "combined" and dataset_mode != name:
            continue
        info = summary.get(name, {})
        if int(info.get("available_entries", 0)) > 0 and info.get("manifest_path"):
            chosen.append(Path(info["manifest_path"]))
    if not chosen:
        raise RuntimeError(f"No usable manifests found for dataset_mode={dataset_mode}.")
    return chosen


def _build_eval_manifest(summary: dict, dataset_mode: str, subset: str, output_dir: Path, max_samples: int):
    selected = []
    for manifest_path in _choose_manifests(summary, dataset_mode):
        entries = _load_jsonl(manifest_path)
        selected.extend([item for item in entries if item.get("split", "train") == subset])
    selected = selected[:max_samples]
    manifest_path = output_dir / f"stage5_{subset}_eval_manifest.jsonl"
    _write_jsonl(manifest_path, selected)
    return manifest_path, selected


def _build_dataset(manifest_path: Path, subset: str, num_points: int, max_samples: int):
    class Config:
        MANIFEST_PATH = str(manifest_path)
        POINTS = num_points
        MAX_SAMPLES = max_samples
        pass

    config = Config()
    config.subset = "all"
    return Stage4PointCacheDataset(config)


def _compute_metrics(model, dataloader, device):
    losses = []
    cosines = []
    angles = []
    sample_keys = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            points = batch["points"].to(device)
            priors = batch["prior_vector"].to(device)
            target_direction = torch.nn.functional.normalize(batch["target_direction"].to(device), dim=-1)
            label_confidence = batch["label_confidence"].to(device).squeeze(-1)
            pred_direction = torch.nn.functional.normalize(model(points, batch["instruction"], priors), dim=-1)
            cosine = torch.nn.functional.cosine_similarity(pred_direction, target_direction, dim=-1)
            cosine_loss = (1 - cosine)
            l1_loss = torch.nn.functional.smooth_l1_loss(
                pred_direction,
                target_direction,
                reduction="none",
            ).mean(dim=-1)
            loss = (0.7 * cosine_loss + 0.3 * l1_loss) * label_confidence
            angle = torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0)))

            losses.extend(loss.detach().cpu().tolist())
            cosines.extend(cosine.detach().cpu().tolist())
            angles.extend(angle.detach().cpu().tolist())
            sample_keys.extend([item.get("sample_key") for item in batch["meta"]])

    return {
        "weighted_loss": round(float(np.mean(losses)) if losses else 0.0, 6),
        "mean_cosine": round(float(np.mean(cosines)) if cosines else 0.0, 6),
        "mean_angle_deg": round(float(np.mean(angles)) if angles else 0.0, 4),
        "median_angle_deg": round(float(np.median(angles)) if angles else 0.0, 4),
        "sample_keys": sample_keys[:10],
    }


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = build_stage5_smoke_manifests(repo_root=repo_root, output_dir=output_dir)
    eval_manifest, entries = _build_eval_manifest(
        summaries,
        args.dataset_mode,
        args.subset,
        output_dir,
        args.max_samples,
    )
    dataset = _build_dataset(eval_manifest, "all", args.num_points, args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError(f"No unseen samples found for subset={args.subset}.")

    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, max(1, len(dataset))),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage5_collate_fn,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (output_dir / "stage5_pilot_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config") or {
        "point_dim": 6,
        "prior_dim": 8,
        "hidden_dim": 128,
        "text_dim": 64,
        "vocab_size": 4096,
        "max_tokens": 32,
    }
    model = PartConditionedOrientationHead(type("Cfg", (), model_config)()).to(device)
    model.load_state_dict(checkpoint["model"])

    metrics = _compute_metrics(model, dataloader, device)
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_mode": args.dataset_mode,
        "subset": args.subset,
        "device": device,
        "manifest_path": str(eval_manifest),
        "dataset_size": len(dataset),
        "metrics": metrics,
    }
    summary_path = output_dir / f"stage5_{args.subset}_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
