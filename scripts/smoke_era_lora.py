#!/usr/bin/env python3
"""Before/after smoke: prompt-only vs era LoRA on one selfie.

Saves PNGs under a chosen --out-dir (default: repo outputs/era_lora_smoke).
Also copies to Project store media when --store-media is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image

from app.generation import (
    DEFAULT_GUIDANCE,
    DEFAULT_LORA_SCALE,
    DEFAULT_STRENGTH,
    PortraitGenerator,
    list_era_loras,
    pick_device,
    resolve_era_lora_path,
)
from data.era_emotion_matcher import load_catalog, match_era_emotion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO / "test_happy_face.jpg",
    )
    parser.add_argument(
        "--era",
        default=None,
        help="Force era (default: matcher result). e.g. 'Modern/Impressionist'",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--lora-scale", type=float, default=DEFAULT_LORA_SCALE)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs" / "era_lora_smoke",
    )
    parser.add_argument(
        "--store-media",
        type=Path,
        default=None,
        help="Also copy before/after into this folder (Project media/era-lora)",
    )
    parser.add_argument(
        "--store-prefix",
        default=None,
        help="Prefix for store filenames, e.g. renaissance → renaissance_before.png",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"missing image: {args.image}", file=sys.stderr)
        return 1

    catalog = load_catalog()
    match = match_era_emotion(image=args.image, catalog=catalog, top_k=3)
    era = args.era or match.era
    prompt = match.style_prompt
    # If forcing an era different from match, still use matcher prompt unless Modern.
    if args.era and args.era != match.era:
        prompt = (
            f"{args.era} oil portrait painting of a person, "
            f"museum lighting, detailed brushwork, {match.emotion.dominant_emotion} expression"
        )

    era_path = resolve_era_lora_path(era)
    print(f"device={pick_device()}")
    print(f"matched_era={match.era} using_era={era}")
    print(f"era_lora={era_path}")
    print(f"available={list_era_loras()}")
    print(
        f"strength={args.strength} lora_scale={args.lora_scale} "
        f"guidance={args.guidance} steps={args.steps}"
    )
    print(f"prompt={prompt[:120]}...")

    if era_path is None:
        print(
            "ERROR: no era LoRA checkpoint found. Train first:\n"
            "  python scripts/train_lora_era.py --era modern_impressionist --max-steps 300",
            file=sys.stderr,
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selfie = Image.open(args.image)
    gen = PortraitGenerator(allow_placeholder=False, lora_scale=args.lora_scale)

    before = gen.generate(
        selfie,
        prompt,
        era=era,
        disable_lora=True,
        strength=args.strength,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        use_faceid=False,
    )
    before_path = args.out_dir / "before_prompt_only.png"
    before.image.save(before_path)

    after = gen.generate(
        selfie,
        prompt,
        era=era,
        disable_lora=False,
        strength=args.strength,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        use_faceid=False,
    )
    after_path = args.out_dir / "after_era_lora.png"
    after.image.save(after_path)

    meta = {
        "era": era,
        "era_lora": str(era_path),
        "prompt": prompt,
        "strength": args.strength,
        "lora_scale": args.lora_scale,
        "guidance": args.guidance,
        "steps": args.steps,
        "match": match.to_dict(),
        "before": before.meta.to_dict(),
        "after": after.meta.to_dict(),
        "before_path": str(before_path),
        "after_path": str(after_path),
    }
    meta_path = args.out_dir / "smoke_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"before": before.meta.to_dict(), "after": after.meta.to_dict()}, indent=2))

    if args.store_media:
        args.store_media.mkdir(parents=True, exist_ok=True)
        prefix = args.store_prefix
        if not prefix and era:
            # Modern/Impressionist → modern
            slug = era.split("/")[0].strip().lower().replace(" ", "_")
            prefix = slug
        named = []
        if prefix:
            named = [
                (before_path, f"{prefix}_before.png"),
                (after_path, f"{prefix}_after.png"),
            ]
            for old_name, weak_name in (
                (f"{prefix}_before.png", f"{prefix}_before_weak.png"),
                (f"{prefix}_after.png", f"{prefix}_after_weak.png"),
            ):
                old = args.store_media / old_name
                if old.is_file():
                    weak = args.store_media / weak_name
                    if not weak.exists():
                        shutil.copy2(old, weak)
                        print(f"kept weak prior as {weak}")
        for src, name in [
            (before_path, "before_prompt_only.png"),
            (after_path, "after_era_lora.png"),
            *named,
        ]:
            dst = args.store_media / name
            shutil.copy2(src, dst)
            print(f"copied {dst}")
        (args.store_media / "smoke_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        if prefix:
            (args.store_media / f"{prefix}_smoke_meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )

    if not after.meta.lora_loaded or after.meta.lora_source != "era":
        print("WARNING: after image did not load era LoRA", file=sys.stderr)
        return 3
    print(f"OK saved {before_path} and {after_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
