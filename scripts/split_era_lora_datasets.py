#!/usr/bin/env python3
"""Split the 104-portrait LoRA set into per-era folders + commit-friendly manifest.

Primary eras (enough images for short real training):
  Modern/Impressionist, Baroque, Romantic
Renaissance is optional; Rococo/Medieval are too small for honest solo LoRAs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ERA_SLUGS = {
    "Medieval/Gothic": "medieval_gothic",
    "Renaissance": "renaissance",
    "Baroque": "baroque",
    "Rococo": "rococo",
    "Romantic": "romantic",
    "Modern/Impressionist": "modern_impressionist",
}

# Eras we actually intend to train first (skip tiny Medieval).
PRIMARY_ERAS = ("Modern/Impressionist", "Baroque", "Romantic")
OPTIONAL_ERAS = ("Renaissance",)


def slugify(era: str) -> str:
    return ERA_SLUGS.get(era, era.lower().replace("/", "_").replace(" ", "_"))


def era_trigger(era: str) -> str:
    """Short trigger phrase prepended to captions for era conditioning."""
    return {
        "Modern/Impressionist": "modern impressionist portrait painting",
        "Baroque": "baroque portrait painting, chiaroscuro",
        "Romantic": "romantic era portrait painting",
        "Renaissance": "renaissance portrait painting",
        "Rococo": "rococo portrait painting",
        "Medieval/Gothic": "medieval gothic portrait",
    }.get(era, f"{era.lower()} portrait painting")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO / "data" / "labels" / "portraits_104_era_emotion.csv",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=REPO / "data" / "lora_training_data" / "images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "lora_training_data" / "by_era",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "data" / "labels" / "era_lora_splits.json",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of hardlinking/symlinking",
    )
    args = parser.parse_args()

    with args.labels.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = Counter(r["era"] for r in rows)
    print("Era counts (labeled 104):")
    for era, n in counts.most_common():
        flag = ""
        if era in PRIMARY_ERAS:
            flag = "  ← primary"
        elif era in OPTIONAL_ERAS:
            flag = "  ← optional"
        elif n < 10:
            flag = "  ← skip (too few)"
        print(f"  {era:24s} {n:3d}{flag}")

    captions: dict[str, str] = {}
    for txt in args.images.glob("*.txt"):
        captions[txt.stem + ".jpg"] = txt.read_text(encoding="utf-8").strip()

    by_era: dict[str, list[dict]] = {}
    for row in rows:
        era = row["era"]
        fname = row["filename"]
        src = args.images / fname
        if not src.is_file():
            print(f"missing image: {src}")
            continue
        slug = slugify(era)
        era_dir = args.out / slug
        era_dir.mkdir(parents=True, exist_ok=True)
        dst = era_dir / fname
        if not dst.exists():
            if args.copy:
                shutil.copy2(src, dst)
            else:
                try:
                    os_link = getattr(src, "hardlink_to", None)
                    if os_link:
                        dst.hardlink_to(src)
                    else:
                        import os

                        os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
        base_cap = captions.get(fname, "portrait painting")
        cap = f"{era_trigger(era)}, {base_cap}"
        (era_dir / f"{Path(fname).stem}.txt").write_text(cap + "\n", encoding="utf-8")
        by_era.setdefault(era, []).append(
            {
                "filename": fname,
                "slug": slug,
                "dominant_emotion": row.get("dominant_emotion"),
                "caption": cap,
            }
        )

    manifest = {
        "source_labels": str(args.labels.relative_to(REPO)),
        "source_images": str(args.images.relative_to(REPO)),
        "output_dir": str(args.out.relative_to(REPO)),
        "counts": dict(counts),
        "primary_eras": list(PRIMARY_ERAS),
        "optional_eras": list(OPTIONAL_ERAS),
        "skipped_note": "Medieval/Gothic (1) and Rococo (8) skipped for solo LoRA; too few images.",
        "eras": {
            era: {
                "slug": slugify(era),
                "n": len(items),
                "train_recommended": era in PRIMARY_ERAS
                or (era in OPTIONAL_ERAS and len(items) >= 12),
                "files": [it["filename"] for it in items],
            }
            for era, items in sorted(by_era.items(), key=lambda kv: -len(kv[1]))
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote manifest: {args.manifest}")
    print(f"Per-era folders under: {args.out}")
    for era in PRIMARY_ERAS:
        slug = slugify(era)
        n = len(list((args.out / slug).glob("*.jpg")))
        print(f"  {slug}: {n} jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
