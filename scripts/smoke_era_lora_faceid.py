#!/usr/bin/env python3
"""FaceID + era LoRA before/after smoke on a fixed selfie.

Writes (overwrite) into the Project media/era-lora store:
  selfie_original.png
  {renaissance,baroque,romantic,modern}_{before,after}.png

before = prompt-only + FaceID (LoRA off)
after  = era LoRA + FaceID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image

from app.generation import (
    DEFAULT_FACEID_SCALE,
    DEFAULT_GUIDANCE,
    DEFAULT_LORA_SCALE,
    PortraitGenerator,
    faceid_status,
    list_era_loras,
    pick_device,
    resolve_era_lora_path,
)
from data.era_emotion_matcher import build_style_prompt, load_catalog, match_era_emotion

ERAS = (
    ("Renaissance", "renaissance"),
    ("Baroque", "baroque"),
    ("Romantic", "romantic"),
    ("Modern/Impressionist", "modern"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfie", type=Path, required=True)
    parser.add_argument("--store-media", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs" / "era_lora_smoke" / "faceid",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=0.72)
    parser.add_argument("--lora-scale", type=float, default=DEFAULT_LORA_SCALE)
    parser.add_argument("--faceid-scale", type=float, default=DEFAULT_FACEID_SCALE)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--retry-strength",
        type=float,
        default=0.65,
        help="If identity drifts on Modern, retry after at this strength",
    )
    args = parser.parse_args()

    if not args.selfie.is_file():
        print(f"missing selfie: {args.selfie}", file=sys.stderr)
        return 1

    ok, reason = faceid_status()
    print(f"faceid_status ok={ok} reason={reason}")
    if not ok:
        print("ERROR: FaceID not available — refusing to fake it", file=sys.stderr)
        return 2

    args.store_media.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selfie_img = Image.open(args.selfie).convert("RGB")
    selfie_store = args.store_media / "selfie_original.png"
    selfie_img.save(selfie_store, format="PNG")
    selfie_img.save(args.out_dir / "selfie_original.png", format="PNG")
    selfie_meta = file_info(selfie_store)
    print(f"selfie_original={selfie_meta}")

    catalog = load_catalog()
    match = match_era_emotion(image=args.selfie, catalog=catalog, top_k=3)
    emotion = match.emotion.dominant_emotion
    print(f"device={pick_device()} seed={args.seed} emotion={emotion}")
    print(f"available_loras={list_era_loras()}")
    print(
        f"strength={args.strength} lora_scale={args.lora_scale} "
        f"faceid_scale={args.faceid_scale} guidance={args.guidance} steps={args.steps}"
    )

    gen = PortraitGenerator(
        allow_placeholder=False,
        lora_scale=args.lora_scale,
        faceid_scale=args.faceid_scale,
    )
    batch: dict = {
        "faceid_smoke": True,
        "selfie": selfie_meta,
        "seed": args.seed,
        "strength": args.strength,
        "lora_scale": args.lora_scale,
        "faceid_scale": args.faceid_scale,
        "guidance": args.guidance,
        "steps": args.steps,
        "emotion": emotion,
        "faceid": reason,
        "device": pick_device(),
        "eras": {},
    }

    for era_label, prefix in ERAS:
        era_path = resolve_era_lora_path(era_label)
        if era_path is None:
            print(f"ERROR: missing LoRA for {era_label}", file=sys.stderr)
            return 3
        prompt = build_style_prompt(era_label, emotion)
        print(f"\n=== {era_label} ({prefix}) ===")
        print(f"prompt={prompt[:140]}...")
        print(f"lora={era_path}")

        strength = (
            args.retry_strength if prefix == "modern" else args.strength
        )

        before = gen.generate(
            selfie_img,
            prompt,
            era=era_label,
            disable_lora=True,
            strength=strength,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            use_faceid=True,
            faceid_scale=args.faceid_scale,
        )
        after = gen.generate(
            selfie_img,
            prompt,
            era=era_label,
            disable_lora=False,
            strength=strength,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            use_faceid=True,
            faceid_scale=args.faceid_scale,
        )

        if not before.meta.faceid_used or not after.meta.faceid_used:
            print(
                f"ERROR: faceid_used false before={before.meta.faceid_used} "
                f"after={after.meta.faceid_used} reason={after.meta.faceid_reason}",
                file=sys.stderr,
            )
            return 4
        if not after.meta.lora_loaded or after.meta.lora_source != "era":
            print(f"ERROR: {era_label} after did not load era LoRA", file=sys.stderr)
            return 5

        before_path = args.store_media / f"{prefix}_before.png"
        after_path = args.store_media / f"{prefix}_after.png"
        before.image.save(args.out_dir / f"{prefix}_before.png")
        after.image.save(args.out_dir / f"{prefix}_after.png")
        before.image.save(before_path)
        after.image.save(after_path)

        entry = {
            "era": era_label,
            "prefix": prefix,
            "prompt": prompt,
            "seed": args.seed,
            "strength": strength,
            "lora_scale": args.lora_scale,
            "faceid_scale": args.faceid_scale,
            "lora_path": str(era_path.resolve()),
            "selfie_path": str(selfie_store.resolve()),
            "selfie_sha256": selfie_meta["sha256"],
            "before": {**file_info(before_path), "meta": before.meta.to_dict()},
            "after": {**file_info(after_path), "meta": after.meta.to_dict()},
        }
        batch["eras"][prefix] = entry
        print(
            f"wrote {before_path.name} faceid={before.meta.faceid_used} "
            f"/ {after_path.name} faceid={after.meta.faceid_used} "
            f"lora={after.meta.lora_source}"
        )

    meta_path = args.store_media / "faceid_smoke_meta.json"
    meta_path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "faceid_smoke_meta.json").write_text(
        json.dumps(batch, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nmeta={meta_path}")
    print("FACEID SMOKE DONE — visually confirm afters match selfie identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
