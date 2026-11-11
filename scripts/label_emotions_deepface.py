#!/usr/bin/env python3
"""Batch-label portrait emotions with DeepFace (M1).

Input:  data/lora_training_data/images/*.jpg
Output: data/labels/portraits_104_emotions.csv
        data/labels/logs/deepface_failures.log
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

EMOTION_KEYS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# Paintings often trip strict detectors; keep enforce_detection=False.
DEFAULT_DETECTOR = "opencv"
FALLBACK_DETECTORS = ("retinaface",)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="DeepFace emotion labels for portrait images")
    p.add_argument(
        "--images-dir",
        type=Path,
        default=root / "data" / "lora_training_data" / "images",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=root / "data" / "labels" / "portraits_104_emotions.csv",
    )
    p.add_argument(
        "--failure-log",
        type=Path,
        default=root / "data" / "labels" / "logs" / "deepface_failures.log",
    )
    p.add_argument("--detector-backend", default=DEFAULT_DETECTOR)
    p.add_argument(
        "--fallback-detectors",
        default=",".join(FALLBACK_DETECTORS),
        help="Comma-separated backends to try when primary yields conf<=0",
    )
    p.add_argument("--glob", default="*.jpg", dest="glob_pattern")
    return p.parse_args()


def setup_logging(failure_log: Path) -> logging.Logger:
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("label_emotions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(failure_log, mode="w", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _run_analyze(path: Path, detector_backend: str) -> dict:
    from deepface import DeepFace

    result = DeepFace.analyze(
        img_path=str(path),
        actions=["emotion"],
        enforce_detection=False,
        detector_backend=detector_backend,
        silent=True,
    )
    if isinstance(result, list):
        result = result[0]
    return result


def _confidence(result: dict) -> float | None:
    face_confidence = result.get("face_confidence")
    if face_confidence is None:
        return None
    try:
        return float(face_confidence)
    except (TypeError, ValueError):
        return None


def analyze_one(path: Path, detector_backend: str, fallbacks: list[str]) -> dict:
    used = detector_backend
    result = _run_analyze(path, detector_backend)
    conf = _confidence(result)
    notes_parts: list[str] = []

    # OpenCV often returns conf=0 + full-frame region on paintings; retry.
    if conf is None or conf <= 0:
        notes_parts.append(f"primary_{detector_backend}_conf_le_0")
        for fb in fallbacks:
            if fb == detector_backend:
                continue
            try:
                alt = _run_analyze(path, fb)
                alt_conf = _confidence(alt)
                if alt_conf is not None and alt_conf > 0:
                    result = alt
                    conf = alt_conf
                    used = fb
                    notes_parts.append(f"fallback_{fb}_ok")
                    break
                notes_parts.append(f"fallback_{fb}_conf_le_0")
            except Exception as exc:  # noqa: BLE001
                notes_parts.append(f"fallback_{fb}_err:{type(exc).__name__}")

    emotions = result.get("emotion") or {}
    dominant = str(result.get("dominant_emotion") or "").lower()
    region = result.get("region") or {}
    w = int(region.get("w") or 0)
    h = int(region.get("h") or 0)

    # Strict face+emotion: positive detector confidence.
    face_detected = conf is not None and conf > 0 and w > 0 and h > 0
    if conf is None:
        conf_val: float | str = ""
        notes_parts.append("no_face_confidence")
    else:
        conf_val = conf
        if conf <= 0:
            notes_parts.append("emotion_without_strict_face")

    notes_parts.append(f"detector={used}")

    row = {
        "filename": path.name,
        "dominant_emotion": dominant or "unknown",
        "face_detected": face_detected,
        "confidence": conf_val,
        "notes": "; ".join(notes_parts),
        "region_x": region.get("x", ""),
        "region_y": region.get("y", ""),
        "region_w": w if w else "",
        "region_h": h if h else "",
    }
    for key in EMOTION_KEYS:
        val = emotions.get(key)
        try:
            row[key] = float(val) if val is not None else ""
        except (TypeError, ValueError):
            row[key] = ""
    return row


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.failure_log)

    images = sorted(args.images_dir.glob(args.glob_pattern))
    if not images:
        logger.error("No images matching %s in %s", args.glob_pattern, args.images_dir)
        return 1

    fallbacks = [b.strip() for b in args.fallback_detectors.split(",") if b.strip()]
    logger.info("Found %d images in %s", len(images), args.images_dir)
    logger.info(
        "detector_backend=%s fallbacks=%s enforce_detection=False",
        args.detector_backend,
        fallbacks,
    )

    fieldnames = [
        "filename",
        "dominant_emotion",
        *EMOTION_KEYS,
        "face_detected",
        "confidence",
        "notes",
        "region_x",
        "region_y",
        "region_w",
        "region_h",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    emotion_ok = 0
    face_emotion_ok = 0
    failures = 0
    t0 = time.time()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(images, start=1):
            try:
                row = analyze_one(path, args.detector_backend, fallbacks)
                has_scores = any(isinstance(row[k], float) for k in EMOTION_KEYS)
                if has_scores and row["dominant_emotion"] not in ("", "unknown"):
                    emotion_ok += 1
                    if row["face_detected"] is True:
                        face_emotion_ok += 1
                    else:
                        logger.warning(
                            "[%d/%d] %s: emotion ok but face_detected=False (conf=%s) notes=%s",
                            i,
                            len(images),
                            path.name,
                            row["confidence"],
                            row["notes"],
                        )
                else:
                    failures += 1
                    row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "no_emotion"
                    logger.warning("[%d/%d] %s: no usable emotion", i, len(images), path.name)
                writer.writerow(row)
                rows.append(row)
                if i % 10 == 0 or i == len(images):
                    elapsed = time.time() - t0
                    logger.info(
                        "Progress %d/%d (%.1fs) emotion_ok=%d face+emotion=%d fail=%d",
                        i,
                        len(images),
                        elapsed,
                        emotion_ok,
                        face_emotion_ok,
                        failures,
                    )
            except Exception as exc:  # noqa: BLE001 — log and continue batch
                failures += 1
                err = f"{type(exc).__name__}: {exc}"
                logger.warning("[%d/%d] %s FAILED: %s", i, len(images), path.name, err)
                empty = {k: "" for k in fieldnames}
                empty.update(
                    {
                        "filename": path.name,
                        "dominant_emotion": "error",
                        "face_detected": False,
                        "notes": err,
                    }
                )
                writer.writerow(empty)
                rows.append(empty)

    total = len(images)
    emo_rate = (emotion_ok / total * 100.0) if total else 0.0
    face_rate = (face_emotion_ok / total * 100.0) if total else 0.0
    logger.info("=" * 60)
    logger.info("Done. wrote %s", args.output_csv)
    logger.info("Total=%d emotion_ok=%d (%.1f%%) failures=%d", total, emotion_ok, emo_rate, failures)
    logger.info("face+emotion (conf>0)=%d (%.1f%%)", face_emotion_ok, face_rate)
    logger.info(
        "Target ≥70%% face+emotion: %s",
        "PASS" if face_rate >= 70 else "BELOW TARGET",
    )
    logger.info("Failure log: %s", args.failure_log)
    return 0 if face_rate >= 70 else 2


if __name__ == "__main__":
    raise SystemExit(main())
