#!/usr/bin/env python3
"""CLI for Phase C / M3 era×emotion matching.

Examples:
  ./venv/bin/python scripts/match_era_emotion.py --image test_happy_face.jpg
  ./venv/bin/python scripts/match_era_emotion.py --demo
  ./venv/bin/python scripts/match_era_emotion.py --dominant happy --log data/labels/logs/era_emotion_match_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.era_emotion_matcher import (  # noqa: E402
    EMOTION_KEYS,
    MODULE_VERSION,
    default_images_dir,
    default_labels_csv,
    match_era_emotion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selfie → era×emotion portrait matcher")
    p.add_argument("--image", type=Path, action="append", default=None, help="Selfie path (repeatable)")
    p.add_argument(
        "--scores",
        type=str,
        default=None,
        help='JSON emotion scores, e.g. \'{"happy":80,"neutral":20}\'',
    )
    p.add_argument("--dominant", type=str, default=None, help="Force/fixture dominant emotion")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--labels-csv", type=Path, default=default_labels_csv())
    p.add_argument("--images-dir", type=Path, default=default_images_dir())
    p.add_argument("--detector-backend", default="opencv")
    p.add_argument(
        "--log",
        type=Path,
        default=ROOT / "data" / "labels" / "logs" / "era_emotion_match_results.jsonl",
    )
    p.add_argument("--no-log", action="store_true", help="Skip writing JSONL log")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run happy/sad sample images + neutral score fixture (reproducible demo)",
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout")
    return p.parse_args()


def _parse_scores(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--scores must be a JSON object")
    return {str(k).lower(): float(v) for k, v in data.items()}


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_one(
    *,
    image: Path | None,
    scores: dict[str, float] | None,
    dominant: str | None,
    args: argparse.Namespace,
    label: str,
) -> dict:
    result = match_era_emotion(
        image=str(image) if image else None,
        scores=scores,
        dominant_emotion=dominant,
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        top_k=args.top_k,
        detector_backend=args.detector_backend,
    )
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "input_image": str(image.resolve()) if image else None,
        "input_scores": scores,
        "input_dominant": dominant,
        "module_version": MODULE_VERSION,
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "result": result.to_dict(),
    }
    return record


def _demo_jobs(root: Path) -> list[dict]:
    """Happy/sad/neutral coverage: live sample when available + score fixtures.

    ``test_sad_face.jpg`` is often read as neutral by DeepFace, so a sad
    score fixture guarantees the melancholy matching path is exercised.
    """
    happy = root / "test_happy_face.jpg"
    sad = root / "test_sad_face.jpg"
    jobs: list[dict] = []
    if happy.is_file():
        jobs.append({"label": "sample_happy", "image": happy, "scores": None, "dominant": None})
    if sad.is_file():
        jobs.append({"label": "sample_sad_image", "image": sad, "scores": None, "dominant": None})

    jobs.append(
        {
            "label": "fixture_sad",
            "image": None,
            "scores": {
                "angry": 4.0,
                "disgust": 1.0,
                "fear": 8.0,
                "happy": 2.0,
                "sad": 72.0,
                "surprise": 1.0,
                "neutral": 12.0,
            },
            "dominant": "sad",
        }
    )
    jobs.append(
        {
            "label": "fixture_neutral",
            "image": None,
            "scores": {
                "angry": 2.0,
                "disgust": 1.0,
                "fear": 3.0,
                "happy": 8.0,
                "sad": 12.0,
                "surprise": 4.0,
                "neutral": 70.0,
            },
            "dominant": "neutral",
        }
    )
    return jobs


def main() -> int:
    args = parse_args()
    jobs: list[dict] = []

    if args.demo:
        jobs.extend(_demo_jobs(ROOT))
    else:
        images = args.image or []
        scores = _parse_scores(args.scores)
        if not images and scores is None and not args.dominant:
            print("Provide --image, --scores/--dominant, or --demo", file=sys.stderr)
            return 2
        if images:
            for img in images:
                jobs.append(
                    {
                        "label": img.name,
                        "image": img,
                        "scores": scores,
                        "dominant": args.dominant,
                    }
                )
        else:
            jobs.append(
                {
                    "label": args.dominant or "scores",
                    "image": None,
                    "scores": scores,
                    "dominant": args.dominant,
                }
            )

    records = []
    for job in jobs:
        if job.get("dominant") and job["dominant"] not in EMOTION_KEYS and job["dominant"] is not None:
            print(f"Unknown dominant emotion: {job['dominant']}", file=sys.stderr)
            return 2
        record = _run_one(
            image=job["image"],
            scores=job["scores"],
            dominant=job["dominant"],
            args=args,
            label=job["label"],
        )
        records.append(record)
        if not args.no_log:
            _append_log(args.log, record)

        res = record["result"]
        refs = ", ".join(r["filename"] for r in res["references"]) or "(none)"
        print(
            f"[{record['label']}] emotion={res['emotion']['dominant_emotion']} "
            f"era={res['era']} mode={res['match_mode']} refs=[{refs}]",
            file=sys.stderr,
        )
        print(f"  why: {res['why_this_era']}", file=sys.stderr)
        print(f"  prompt: {res['style_prompt']}", file=sys.stderr)

    payload = records if len(records) > 1 else records[0]
    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False))

    if not args.no_log:
        print(f"log: {args.log.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
