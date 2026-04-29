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
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--balance-datasets", action="store_true")
    parser.add_argument("--exclude-label-source", action="append", default=[])
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    parser.add_argument("--manifest-path", type=str, default="")
    parser.add_argument("--init-checkpoint", type=str, default="")
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


def _filter_entries(entries, excluded_sources, min_label_confidence):
    excluded_sources = {str(item) for item in (excluded_sources or []) if str(item).strip()}
    filtered = []
    for item in entries:
        source = str(item.get("train_target_direction_source", "") or "")
        confidence = float(item.get("train_label_confidence", 0.0) or 0.0)
        if source in excluded_sources:
            continue
        if confidence < float(min_label_confidence):
            continue
        filtered.append(item)
    return filtered


def _balance_entries(entries, total_cap: int):
    dataset_groups = {}
    for item in entries:
        dataset_groups.setdefault(str(item.get("dataset", "unknown")), []).append(item)

    if len(dataset_groups) <= 1:
        return entries[:total_cap] if total_cap > 0 else entries

    if total_cap > 0 and total_cap < len(dataset_groups):
        raise RuntimeError(
            f"Unable to balance {len(dataset_groups)} datasets with total_cap={total_cap}."
        )

    per_dataset_cap = min(len(group) for group in dataset_groups.values())
    if total_cap > 0:
        per_dataset_cap = min(per_dataset_cap, total_cap // len(dataset_groups))

    balanced = []
    for dataset_name in sorted(dataset_groups.keys()):
        balanced.extend(dataset_groups[dataset_name][:per_dataset_cap])
    return balanced


def _entry_stats(entries):
    dataset_counts = {}
    label_source_counts = {}
    confidences = []

    for item in entries:
        dataset = str(item.get("dataset", "unknown"))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        source = str(item.get("train_target_direction_source", "unknown"))
        label_source_counts[source] = label_source_counts.get(source, 0) + 1

        confidence = float(item.get("train_label_confidence", 0.0) or 0.0)
        confidences.append(confidence)

    return {
        "count": len(entries),
        "dataset_counts": dataset_counts,
        "label_source_counts": label_source_counts,
        "mean_label_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 6),
    }


def _prepare_train_val_manifests(
    summary: dict,
    dataset_mode: str,
    output_dir: Path,
    max_train: int,
    max_val: int,
    max_test: int,
    balance_datasets: bool = False,
    excluded_label_sources=None,
    min_label_confidence: float = 0.0,
):
    manifests = _choose_manifests(summary, dataset_mode)
    train_entries = []
    val_entries = []
    test_entries = []

    for dataset_name, manifest_path in manifests:
        entries = _load_manifest_entries(manifest_path)
        train_part = [item for item in entries if item.get("split", "train") == "train"]
        val_part = [item for item in entries if item.get("split", "train") == "val"]
        test_part = [item for item in entries if item.get("split", "train") == "test"]
        train_entries.extend(train_part)
        val_entries.extend(val_part)
        test_entries.extend(test_part)

    train_entries = _filter_entries(train_entries, excluded_label_sources, min_label_confidence)
    val_entries = _filter_entries(val_entries, excluded_label_sources, min_label_confidence)
    test_entries = _filter_entries(test_entries, excluded_label_sources, min_label_confidence)

    if not val_entries and len(train_entries) > 1:
        val_entries = train_entries[-1:]
        train_entries = train_entries[:-1]

    if dataset_mode == "combined" and balance_datasets:
        train_entries = _balance_entries(train_entries, max_train)
        val_entries = _balance_entries(val_entries, max_val)
        test_entries = _balance_entries(test_entries, max_test)
    else:
        train_entries = train_entries[:max_train]
        val_entries = val_entries[:max_val]
        test_entries = test_entries[:max_test]

    train_manifest = output_dir / "stage5_train_manifest.jsonl"
    val_manifest = output_dir / "stage5_val_manifest.jsonl"
    test_manifest = output_dir / "stage5_test_manifest.jsonl"
    _write_jsonl(train_manifest, train_entries)
    _write_jsonl(val_manifest, val_entries)
    _write_jsonl(test_manifest, test_entries)
    return (
        train_manifest,
        val_manifest,
        test_manifest,
        train_entries,
        val_entries,
        test_entries,
        {
            "train": _entry_stats(train_entries),
            "val": _entry_stats(val_entries),
            "test": _entry_stats(test_entries),
        },
    )


def _build_dataset(manifest_path: Path, subset: str, num_points: int, max_samples: int):
    config = SimpleNamespace(
        MANIFEST_PATH=str(manifest_path),
        POINTS=num_points,
        subset=subset,
        MAX_SAMPLES=max_samples,
    )
    return Stage4PointCacheDataset(config)


def _compute_weighted_direction_loss(pred_direction, target_direction, label_confidence):
    pred_direction = torch.nn.functional.normalize(pred_direction, dim=-1)
    target_direction = torch.nn.functional.normalize(target_direction, dim=-1)
    cosine = torch.nn.functional.cosine_similarity(pred_direction, target_direction, dim=-1)
    cosine_loss = (1 - cosine)
    l1_loss = torch.nn.functional.smooth_l1_loss(
        pred_direction,
        target_direction,
        reduction="none",
    ).mean(dim=-1)
    combined = 0.7 * cosine_loss + 0.3 * l1_loss
    weighted = combined * label_confidence
    return weighted.mean(), cosine


def _finite_or_raise(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite values detected in {name}")


def _evaluate(model, dataloader, device):
    model.eval()
    losses = []
    cosine_scores = []
    with torch.no_grad():
        for batch in dataloader:
            points = batch["points"].to(device)
            priors = batch["prior_vector"].to(device)
            target_direction = batch["target_direction"].to(device)
            label_confidence = batch["label_confidence"].to(device).squeeze(-1)
            pred_direction = model(points, batch["instruction"], priors)
            loss, cosine = _compute_weighted_direction_loss(
                pred_direction,
                target_direction,
                label_confidence,
            )
            losses.append(float(loss.item()))
            cosine_scores.extend([float(x) for x in cosine.detach().cpu().tolist()])
    return {
        "loss": round(float(np.mean(losses)) if losses else 0.0, 6),
        "mean_cosine": round(float(np.mean(cosine_scores)) if cosine_scores else 0.0, 6),
    }


def _load_init_checkpoint(model, checkpoint_path: Path, device: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing init checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model")
    if not state_dict:
        raise RuntimeError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return payload


def main():
    args = parse_args()
    set_seed(args.seed)

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    max_test_samples = args.max_test_samples if args.max_test_samples > 0 else args.max_val_samples

    summaries = build_stage5_smoke_manifests(repo_root=repo_root, output_dir=output_dir)
    custom_manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else None
    (
        train_manifest,
        val_manifest,
        test_manifest,
        train_entries,
        val_entries,
        test_entries,
        manifest_stats,
    ) = _prepare_train_val_manifests(
        summaries,
        args.dataset_mode,
        output_dir,
        args.max_train_samples,
        args.max_val_samples,
        max_test_samples,
        args.balance_datasets,
        args.exclude_label_source,
        args.min_label_confidence,
    )

    if custom_manifest_path:
        custom_entries = _load_manifest_entries(custom_manifest_path)
        custom_entries = _filter_entries(custom_entries, args.exclude_label_source, args.min_label_confidence)
        train_entries = [e for e in custom_entries if e.get("split") == "train"]
        val_entries = [e for e in custom_entries if e.get("split") == "val"]
        test_entries = [e for e in custom_entries if e.get("split") == "test"]
        if not val_entries and len(train_entries) > 1:
            val_entries = train_entries[-1:]
            train_entries = train_entries[:-1]
        train_entries = train_entries[:args.max_train_samples]
        val_entries = val_entries[:args.max_val_samples]
        test_entries = test_entries[:max_test_samples]
        train_manifest = output_dir / "stage5_train_manifest.jsonl"
        val_manifest = output_dir / "stage5_val_manifest.jsonl"
        test_manifest = output_dir / "stage5_test_manifest.jsonl"
        _write_jsonl(train_manifest, train_entries)
        _write_jsonl(val_manifest, val_entries)
        _write_jsonl(test_manifest, test_entries)
        manifest_stats = {
            "train": _entry_stats(train_entries),
            "val": _entry_stats(val_entries),
            "test": _entry_stats(test_entries),
        }

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
    init_payload = None
    if args.init_checkpoint:
        init_payload = _load_init_checkpoint(model, Path(args.init_checkpoint), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_logs = []
    best_val_loss = None
    best_ckpt_path = output_dir / "stage5_pilot_best.pth"
    last_ckpt_path = output_dir / "stage5_pilot_last.pth"
    skipped_bad_batches = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            points = batch["points"].to(device)
            priors = batch["prior_vector"].to(device)
            target_direction = batch["target_direction"].to(device)
            label_confidence = batch["label_confidence"].to(device).squeeze(-1)

            optimizer.zero_grad(set_to_none=True)
            try:
                _finite_or_raise("points", points)
                _finite_or_raise("priors", priors)
                _finite_or_raise("target_direction", target_direction)
                _finite_or_raise("label_confidence", label_confidence)
                pred_direction = model(points, batch["instruction"], priors)
                _finite_or_raise("pred_direction", pred_direction)
                loss, _ = _compute_weighted_direction_loss(
                    pred_direction,
                    target_direction,
                    label_confidence,
                )
                _finite_or_raise("loss", loss)
            except ValueError:
                skipped_bad_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = round(float(np.mean(train_losses)), 6) if train_losses else 0.0
        val_metrics = _evaluate(model, val_loader, device) if val_loader is not None else None
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_mean_cosine": None if val_metrics is None else val_metrics["mean_cosine"],
        }
        epoch_logs.append(epoch_log)

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch_log": epoch_log,
            "model_config": {
                "point_dim": model_config.point_dim,
                "prior_dim": model_config.prior_dim,
                "hidden_dim": model_config.hidden_dim,
                "text_dim": model_config.text_dim,
                "vocab_size": model_config.vocab_size,
                "max_tokens": model_config.max_tokens,
            },
        }
        torch.save(ckpt_payload, last_ckpt_path)
        if val_metrics is not None:
            if best_val_loss is None or val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(ckpt_payload, best_ckpt_path)
        elif best_val_loss is None:
            best_val_loss = train_loss
            torch.save(ckpt_payload, best_ckpt_path)

    if best_ckpt_path.exists():
        best_payload = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_payload["model"])

    test_dataset = _build_dataset(test_manifest, "all", args.num_points, args.max_val_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(args.batch_size, max(1, len(test_dataset))) if len(test_dataset) > 0 else 1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage5_collate_fn,
    ) if len(test_dataset) > 0 else None
    test_metrics = _evaluate(model, test_loader, device) if test_loader is not None else None

    summary = {
        "dataset_mode": args.dataset_mode,
        "device": device,
        "epochs": args.epochs,
        "num_points": args.num_points,
        "batch_size": min(args.batch_size, max(1, len(train_dataset))),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "test_manifest": str(test_manifest),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "best_val_loss": best_val_loss,
        "final_test_metrics": test_metrics,
        "skipped_bad_batches": skipped_bad_batches,
        "sampling_config": {
            "balance_datasets": args.balance_datasets,
            "exclude_label_source": list(args.exclude_label_source),
            "min_label_confidence": float(args.min_label_confidence),
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": max_test_samples,
        },
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else "",
        "init_checkpoint_epoch": None if init_payload is None else init_payload.get("epoch"),
        "manifest_stats": manifest_stats,
        "epoch_logs": epoch_logs,
        "label_definition": {
            "primary": "normalized geometry_priors.part_to_object_vector",
            "fallback_open6dor": "orientation_mode template axis",
            "fallback_default": "[0, 0, 1]",
            "label_weight": "train_label_confidence",
        },
        "loss_definition": {
            "type": "0.7 * weighted_cosine + 0.3 * weighted_smooth_l1",
            "weight_source": "train_label_confidence",
        },
    }

    summary_path = output_dir / "stage5_tiny_train_summary.json"
    summary["summary_path"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
