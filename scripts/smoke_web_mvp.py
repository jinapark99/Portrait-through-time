#!/usr/bin/env python3
"""Smoke-test match + generate without starting uvicorn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image

from app.generation import get_generator, faceid_status, pick_device
from data.era_emotion_matcher import match_era_emotion, load_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO / "test_happy_face.jpg",
        help="Input selfie / face image",
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--strength", type=float, default=0.72)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "outputs" / "smoke_portrait.png",
    )
    parser.add_argument("--match-only", action="store_true")
    args = parser.parse_args()

    if not args.image.is_file():
        # Fall back to a training portrait if test faces missing.
        alt = next((REPO / "data" / "lora_training_data" / "images").glob("*.jpg"), None)
        if alt is None:
            print(f"No input image at {args.image}", file=sys.stderr)
            return 1
        args.image = alt
        print(f"Using fallback image: {args.image}")

    print(f"device={pick_device()} faceid={faceid_status()}")
    catalog = load_catalog()
    match = match_era_emotion(image=args.image, catalog=catalog, top_k=3)
    print(json.dumps(match.to_dict(), indent=2)[:2000])

    if args.match_only:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gen = get_generator()
    result = gen.generate(
        Image.open(args.image),
        match.style_prompt,
        strength=args.strength,
        steps=args.steps,
    )
    result.image.save(args.out)
    print(json.dumps(result.meta.to_dict(), indent=2))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
