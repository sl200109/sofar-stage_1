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
    "stage4_point_cache_dataset_train_module",
    "Stage4PointCacheDataset",
)
stage5_collate_fn = _load_symbol(
    REPO_ROOT / "orientation" / "datasets" / "Stage4PointCache.py",
    "stage4_point_cache_dataset_train_collate_module",
    "stage5_collate_fn",
)
PartConditionedOrientationHead = _load_symbol(
    REPO_ROOT / "orientation" / "models" / "PartConditionedOrientationHead.py",
    "part_conditioned_orientation_head_train_module",
    "PartConditionedOrientationHead",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 5 tiny-train pilot entry.")
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--dataset-mode", choices=["combined", "spatialbench", "open6dor"], default="combined")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _load_manifest_entries(manifest_path: Path):
    entries = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
            chosen.append((name, Path(info["manifest_path"])))
    if not chosen:
        raise RuntimeError(f"No usable manifests found for dataset_mode={dataset_mode}.")
    return chosen


def _prepare_train_val_manifests(summary: dict, dataset_mode: str, output_dir: Path, max_train: int, max_val: int):
    manifests = _choose_manifests(summary, dataset_mode)
    train_entries = []
    val_entries = []

    for dataset_name, manifest_path in manifests:
        entries = _load_manifest_entries(manifest_path)
        train_part = [item for item in entries if item.get("split", "train") == "train"]
        val_part = [item for item in entries if item.get("split", "train") == "val"]
        train_entries.extend(train_part)
        val_entries.extend(val_part)

    if not val_entries and len(train_entries) > 1:
        val_entries = train_entries[-1:]
        train_entries = train_entries[:-1]

    train_entries = train_entries[:max_train]
    val_entries = val_entries[:max_val]

    train_manifest = output_dir / "stage5_train_manifest.jsonl"
    val_manifest = output_dir / "stage5_val_manifest.jsonl"
    _write_jsonl(train_manifest, train_entries)
    _write_jsonl(val_manifest, val_entries)
    return train_manifest, val_manifest, train_entries, val_entries


def _build_dataset(manifest_path: Path, subset: str, num_points: int, max_samples: int):
    config = SimpleNamespace(
        MANIFEST_PATH=str(manifest_path),
        POINTS=num_points,
        subset=subset,
        MAX_SAMPLES=max_samples,
    )
    return Stage4PointCacheDataset(config)


def _evaluate(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            points = batch["points"].to(device)
            priors = batch["prior_vector"].to(device)
            target_direction = batch["target_direction"].to(device)
            label_confidence = batch["label_confidence"].to(device).squeeze(-1)
            pred_direction = model(points, batch["instruction"], priors)
            cosine = torch.nn.functional.cosine_similarity(pred_direction, target_direction, dim=-1)
            loss = ((1 - cosine) * label_confidence).mean()
            losses.append(float(loss.item()))
    return round(float(np.mean(losses)) if losses else 0.0, 6)


def main():
    args = parse_args()
    set_seed(args.seed)

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = build_stage5_smoke_manifests(repo_root=repo_root, output_dir=output_dir)
    train_manifest, val_manifest, train_entries, val_entries = _prepare_train_val_manifests(
        summaries,
        args.dataset_mode,
        output_dir,
        args.max_train_samples,
        args.max_val_samples,
    )

    train_dataset = _build_dataset(train_manifest, "all", args.num_points, args.max_train_samples)
    val_dataset = _build_dataset(val_manifest, "all", args.num_points, args.max_val_samples)
    if len(train_dataset) == 0:
        raise RuntimeError("Tiny-train pilot has no train samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, max(1, len(train_dataset))),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=stage5_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size, max(1, len(val_dataset))) if len(val_dataset) > 0 else 1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage5_collate_fn,
    ) if len(val_dataset) > 0 else None

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_logs = []
    best_val_loss = None
    best_ckpt_path = output_dir / "stage5_pilot_best.pth"
    last_ckpt_path = output_dir / "stage5_pilot_last.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            points = batch["points"].to(device)
            priors = batch["prior_vector"].to(device)
            target_direction = batch["target_direction"].to(device)
            label_confidence = batch["label_confidence"].to(device).squeeze(-1)

            optimizer.zero_grad(set_to_none=True)
            pred_direction = model(points, batch["instruction"], priors)
            cosine = torch.nn.functional.cosine_similarity(pred_direction, target_direction, dim=-1)
            loss = ((1 - cosine) * label_confidence).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = round(float(np.mean(train_losses)), 6) if train_losses else 0.0
        val_loss = _evaluate(model, val_loader, device) if val_loader is not None else None
        epoch_log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        epoch_logs.append(epoch_log)

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch_log": epoch_log,
        }
        torch.save(ckpt_payload, last_ckpt_path)
        if val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt_payload, best_ckpt_path)
        elif best_val_loss is None:
            best_val_loss = train_loss
            torch.save(ckpt_payload, best_ckpt_path)

    summary = {
        "dataset_mode": args.dataset_mode,
        "device": device,
        "epochs": args.epochs,
        "num_points": args.num_points,
        "batch_size": min(args.batch_size, max(1, len(train_dataset))),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "best_val_loss": best_val_loss,
        "epoch_logs": epoch_logs,
        "label_definition": {
            "primary": "normalized geometry_priors.part_to_object_vector",
            "fallback_open6dor": "orientation_mode template axis",
            "fallback_default": "[0, 0, 1]",
            "label_weight": "train_label_confidence",
        },
    }

    summary_path = output_dir / "stage5_tiny_train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
