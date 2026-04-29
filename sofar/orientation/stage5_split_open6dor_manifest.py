import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FAMILY_UPRIGHT = "upright_vertical"
FAMILY_FLAT = "flat_upside_down_lying_flat"
FAMILY_PLUG = "plug_cap_sideways"
FAMILY_UNKNOWN = "unknown"


def _normalize_mode(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def infer_open6dor_task_family(orientation_mode: Any) -> str:
    mode = _normalize_mode(orientation_mode)
    if mode in {"upright", "upright_lens_forth", "vertical"}:
        return FAMILY_UPRIGHT
    if mode in {"lying_flat", "lying_sideways", "upside_down", "upside_down_textual"}:
        return FAMILY_FLAT
    if mode.startswith(("plug_", "cap_", "handle_")) or mode in {
        "clip_sideways",
        "sideways",
        "sideways_textual",
    }:
        return FAMILY_PLUG
    return FAMILY_UNKNOWN


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _write_jsonl(path: Path, entries: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Split Open6DOR Stage 5 manifest into task-family expert manifests.")
    parser.add_argument("--manifest-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--families",
        type=str,
        default="upright_vertical,flat_upside_down_lying_flat,plug_cap_sideways",
        help="Comma-separated family names to export.",
    )
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    parser.add_argument("--exclude-label-source", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    export_families = {
        part.strip()
        for part in str(args.families or "").split(",")
        if part.strip()
    }
    excluded_sources = {
        str(item).strip()
        for item in (args.exclude_label_source or [])
        if str(item).strip()
    }

    entries = _load_jsonl(manifest_path)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    skipped_low_confidence = 0
    skipped_excluded_source = 0

    for entry in entries:
        parser_output = entry.get("parser_output") or {}
        family = infer_open6dor_task_family(parser_output.get("orientation_mode"))
        if family not in export_families:
            continue
        source = str(entry.get("train_target_direction_source", "") or "")
        confidence = float(entry.get("train_label_confidence", 0.0) or 0.0)
        if source in excluded_sources:
            skipped_excluded_source += 1
            continue
        if confidence < float(args.min_label_confidence):
            skipped_low_confidence += 1
            continue
        enriched = dict(entry)
        enriched["task_family"] = family
        grouped.setdefault(family, []).append(enriched)

    summary: Dict[str, Any] = {
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "min_label_confidence": float(args.min_label_confidence),
        "excluded_label_sources": sorted(excluded_sources),
        "skipped_low_confidence": skipped_low_confidence,
        "skipped_excluded_source": skipped_excluded_source,
        "families": {},
    }

    for family in sorted(export_families):
        family_entries = grouped.get(family, [])
        family_dir = output_dir / family
        manifest_out = family_dir / "stage5_manifest.jsonl"
        _write_jsonl(manifest_out, family_entries)

        split_counts: Dict[str, int] = {}
        mode_counts: Dict[str, int] = {}
        for entry in family_entries:
            split = str(entry.get("split", "unknown"))
            split_counts[split] = split_counts.get(split, 0) + 1
            mode = _normalize_mode((entry.get("parser_output") or {}).get("orientation_mode"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        summary["families"][family] = {
            "manifest_path": str(manifest_out),
            "count": len(family_entries),
            "split_counts": split_counts,
            "orientation_mode_counts": mode_counts,
        }

    summary_path = output_dir / "stage5_open6dor_family_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
