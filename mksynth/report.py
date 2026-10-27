"""Combine per-level outputs into one dataset: merged MOD manifests and a statistics table.

    python -m mksynth.report out/full          # expects out/full/{easy,medium,hard}/

Writes out/full/mod_train.json, out/full/mod_val.json (paths relative to out/full),
out/full/stats.json and out/full/STATS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mksynth.layout import LEVELS

PAPER_RATIO = {"easy": 0.0724, "medium": 0.1533, "hard": 0.1347}  # MessyKitchens-Real, paper Table 1


def merge(root: Path) -> dict:
    merged = {"train": [], "val": []}
    stats = {}
    for level in LEVELS:
        level_dir = root / level
        if not (level_dir / "stats.json").is_file():
            continue
        stats[level] = json.loads((level_dir / "stats.json").read_text())
        validation = level_dir / "validation.json"
        if validation.is_file():
            v = json.loads(validation.read_text())
            stats[level]["validation"] = {"scenes": v["scenes"], "failed": v["failed"], "dataset_errors": v["dataset_errors"]}
        for split in ("train", "val"):
            path = level_dir / f"mod_{split}.json"
            if path.is_file():
                for record in json.loads(path.read_text()):
                    merged[split].append({**record, **{key: f"{level}/{record[key]}" for key in
                                                       ("image_path", "instance_mask_path", "mesh_path", "scene_path")}})
    for split, records in merged.items():
        (root / f"mod_{split}.json").write_text(json.dumps(records, indent=1))
    (root / "stats.json").write_text(json.dumps(stats, indent=2))
    (root / "STATS.md").write_text(table(stats, merged))
    return stats


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def table(stats: dict, merged: dict) -> str:
    lines = ["| level / split | scenes | failed | MOD records | objects/scene | assets | scenes with nest | with stack | "
             "pen/contact (pooled) | paper real | max pen. p95 (mm) | layout attempts | physics s | render s | validated |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for level, by_split in stats.items():
        validation = by_split.get("validation")
        for key, s in by_split.items():
            if key == "validation":
                continue
            seconds = s["seconds"]
            lines.append("| " + " | ".join([
                key, str(s["scenes"]), str(s["failed"]), str(s["mod_records"]),
                _fmt(s["objects_per_scene"]["mean"], 1) if s["objects_per_scene"] else "-",
                str(s["distinct_assets"]), str(s["scenes_with_nest"]), str(s["scenes_with_stack"]),
                _fmt(s["penetration_contact_ratio_pooled"]), _fmt(PAPER_RATIO.get(level)),
                _fmt(s["max_penetration_depth_mm"]["p95"], 2) if s["max_penetration_depth_mm"] else "-",
                _fmt(s["physics_attempts"]["mean"], 2) if s["physics_attempts"] else "-",
                _fmt(seconds["physics"]["mean"], 1) if seconds["physics"] else "-",
                _fmt(seconds["render"]["mean"], 1) if seconds["render"] else "-",
                "-" if validation is None else f"{validation['scenes'] - validation['failed']}/{validation['scenes']}",
            ]) + " |")
    lines.append("")
    lines.append(f"MOD records: {len(merged['train'])} train, {len(merged['val'])} val.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    merge(root)
    print((root / "STATS.md").read_text())


if __name__ == "__main__":
    main()
