#!/usr/bin/env python3
"""Join Met/smart_metadata era buckets with DeepFace emotion labels (M2 / Phase B).

Input:
  data/lora_training_data/captions.csv
  data/smart_metadata/smart_portraits_final.csv
  data/labels/portraits_104_emotions.csv

Output:
  data/labels/portraits_104_era_emotion.csv
  data/labels/portraits_104_era_uncertain.csv

Era buckets (midpoint of object_begin_date / object_end_date):
  Medieval/Gothic      < 1400
  Renaissance          1400–1599
  Baroque              1600–1719
  Rococo               1720–1779
  Romantic             1780–1849
  Modern/Impressionist >= 1850
  unlabeled            no usable year (or unmatched metadata)

Deterministic: sorted filename order, stable string keys, no randomness.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ERA_BUCKETS = (
    "Medieval/Gothic",
    "Renaissance",
    "Baroque",
    "Rococo",
    "Romantic",
    "Modern/Impressionist",
    "unlabeled",
)

# Inclusive lower bounds; upper exclusive except Modern open-ended.
ERA_CUTOFFS = (
    (1400, "Medieval/Gothic"),
    (1600, "Renaissance"),
    (1720, "Baroque"),
    (1780, "Rococo"),
    (1850, "Romantic"),
)

# Flag rows whose date span is this wide (years) or wider.
WIDE_SPAN_YEARS = 40
# Flag rows within this many years of an era boundary.
BOUNDARY_MARGIN = 5

EMOTION_COLS = (
    "dominant_emotion",
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
    "face_detected",
    "confidence",
    "notes",
    "region_x",
    "region_y",
    "region_w",
    "region_h",
)

YEAR_IN_TEXT = re.compile(r"(?:1[2-9]|20)\d{2}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Era + emotion join for 104 training portraits")
    p.add_argument(
        "--captions-csv",
        type=Path,
        default=root / "data" / "lora_training_data" / "captions.csv",
    )
    p.add_argument(
        "--smart-metadata-csv",
        type=Path,
        default=root / "data" / "smart_metadata" / "smart_portraits_final.csv",
    )
    p.add_argument(
        "--emotions-csv",
        type=Path,
        default=root / "data" / "labels" / "portraits_104_emotions.csv",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=root / "data" / "labels" / "portraits_104_era_emotion.csv",
    )
    p.add_argument(
        "--uncertain-csv",
        type=Path,
        default=root / "data" / "labels" / "portraits_104_era_uncertain.csv",
    )
    return p.parse_args()


def norm_key(artist: str, title: str) -> tuple[str, str]:
    return artist.strip().lower(), title.strip().lower()


def parse_int_year(value: str | None) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        m = YEAR_IN_TEXT.search(s)
        return int(m.group(0)) if m else None


def year_from_object_date(object_date: str) -> int | None:
    """Fallback: first plausible year in free-text object_date."""
    if not object_date or not str(object_date).strip():
        return None
    years = [int(y) for y in YEAR_IN_TEXT.findall(str(object_date))]
    if not years:
        return None
    # Prefer earliest mentioned production year (ignore late print/cast noise later).
    return min(years)


def midpoint_year(begin: int | None, end: int | None, object_date: str) -> tuple[int | None, str]:
    """Return (year, year_source)."""
    if begin is not None and end is not None:
        return (begin + end) // 2, "begin_end_midpoint"
    if begin is not None:
        return begin, "begin_only"
    if end is not None:
        return end, "end_only"
    y = year_from_object_date(object_date)
    if y is not None:
        return y, "object_date_heuristic"
    return None, "missing"


def era_from_year(year: int | None) -> str:
    if year is None:
        return "unlabeled"
    for cutoff, label in ERA_CUTOFFS:
        if year < cutoff:
            return label
    return "Modern/Impressionist"


def near_boundary(year: int | None) -> str | None:
    if year is None:
        return None
    for cutoff, _ in ERA_CUTOFFS:
        if abs(year - cutoff) <= BOUNDARY_MARGIN:
            return f"near_boundary_{cutoff}"
    return None


def artist_title_era_hint(artist: str, title: str) -> str | None:
    """Optional soft hint used only for uncertainty notes, not primary labeling."""
    text = f"{artist} {title}".lower()
    hints = []
    if any(k in text for k in ("gothic", "medieval", "martini", "giotto")):
        hints.append("Medieval/Gothic")
    if any(k in text for k in ("holbein", "dürer", "durer", "renaissance", "raphael")):
        hints.append("Renaissance")
    if any(k in text for k in ("rembrandt", "baroque", "hollar", "castiglione")):
        hints.append("Baroque")
    if any(k in text for k in ("rococo", "watteau", "fragonard")):
        hints.append("Rococo")
    if any(k in text for k in ("romantic", "delacroix", "sully", "daumier")):
        hints.append("Romantic")
    if any(k in text for k in ("van gogh", "gauguin", "impression", "courbet", "fantin")):
        hints.append("Modern/Impressionist")
    return ",".join(sorted(set(hints))) if hints else None


def load_smart_index(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = norm_key(row.get("artist", ""), row.get("title", ""))
        # First occurrence wins for determinism (CSV order).
        if key not in index:
            index[key] = row
    return index


def load_emotions(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["filename"]: r for r in rows}


def uncertain_reasons(
    *,
    matched: bool,
    begin: int | None,
    end: int | None,
    year: int | None,
    year_source: str,
    era: str,
    artist: str,
    title: str,
) -> list[str]:
    reasons: list[str] = []
    if not matched:
        reasons.append("unmatched_smart_metadata")
    if year is None:
        reasons.append("missing_year")
    if year_source == "object_date_heuristic":
        reasons.append("year_from_object_date_heuristic")
    if begin is not None and end is not None and (end - begin) >= WIDE_SPAN_YEARS:
        reasons.append(f"wide_date_span_{end - begin}y")
    nb = near_boundary(year)
    if nb:
        reasons.append(nb)
    hint = artist_title_era_hint(artist, title)
    if hint and era != "unlabeled" and era not in hint.split(","):
        reasons.append(f"artist_title_hint_differs={hint}")
    # Vague free-text date markers.
    title_l = title.lower()
    if "manner of" in title_l or "school of" in artist.lower() or "anonymous" in artist.lower():
        reasons.append("attribution_uncertain")
    return reasons


def main() -> int:
    args = parse_args()
    for required in (args.captions_csv, args.smart_metadata_csv, args.emotions_csv):
        if not required.is_file():
            print(f"ERROR: missing {required}", file=sys.stderr)
            return 1

    smart = load_smart_index(args.smart_metadata_csv)
    emotions = load_emotions(args.emotions_csv)

    with args.captions_csv.open(newline="", encoding="utf-8") as f:
        captions = list(csv.DictReader(f))

    captions = sorted(captions, key=lambda r: r["filename"])

    out_fields = [
        "filename",
        "artist",
        "title",
        "object_id",
        "object_date",
        "object_begin_date",
        "object_end_date",
        "year_mid",
        "year_source",
        "era",
        "era_confidence",
        "era_uncertain",
        "era_uncertain_reasons",
        "match_source",
        *EMOTION_COLS,
    ]

    rows_out: list[dict] = []
    uncertain_rows: list[dict] = []
    era_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    uncertain_count = 0
    heuristic_fallback_count = 0

    for cap in captions:
        filename = cap["filename"]
        artist = cap.get("artist", "")
        title = cap.get("title", "")
        key = norm_key(artist, title)
        meta = smart.get(key)

        matched = meta is not None
        match_source = "smart_metadata_artist_title" if matched else "unmatched"
        object_id = meta.get("object_id", "") if meta else ""
        object_date = meta.get("object_date", "") if meta else ""
        begin = parse_int_year(meta.get("object_begin_date") if meta else None)
        end = parse_int_year(meta.get("object_end_date") if meta else None)

        year, year_source = midpoint_year(begin, end, object_date)
        era = era_from_year(year)

        # Artist/title heuristic only if still unlabeled after date sources.
        if era == "unlabeled":
            hint = artist_title_era_hint(artist, title)
            if hint and "," not in hint:
                era = hint
                year_source = "artist_title_heuristic"
                heuristic_fallback_count += 1
                match_source = (
                    "artist_title_heuristic"
                    if match_source == "unmatched"
                    else f"{match_source}+artist_title_heuristic"
                )

        reasons = uncertain_reasons(
            matched=matched,
            begin=begin,
            end=end,
            year=year if year_source != "artist_title_heuristic" else None,
            year_source=year_source,
            era=era,
            artist=artist,
            title=title,
        )
        if year_source == "artist_title_heuristic":
            reasons.append("artist_title_heuristic_primary")
        if year_source == "object_date_heuristic":
            heuristic_fallback_count += 1

        is_uncertain = bool(reasons)
        if is_uncertain:
            uncertain_count += 1

        emo = emotions.get(filename)
        if emo is None:
            emo = {c: ("unlabeled" if c == "dominant_emotion" else "") for c in EMOTION_COLS}
            emo["face_detected"] = "False"
            emo["notes"] = "missing_emotion_row"

        era_confidence = "high"
        if era == "unlabeled":
            era_confidence = "none"
        elif is_uncertain:
            era_confidence = "low" if any(
                r.startswith("wide_date_span") or r in ("missing_year", "unmatched_smart_metadata")
                or r.endswith("_heuristic")
                or r.startswith("artist_title_heuristic")
                for r in reasons
            ) else "medium"

        row = {
            "filename": filename,
            "artist": artist,
            "title": title,
            "object_id": object_id,
            "object_date": object_date,
            "object_begin_date": begin if begin is not None else "",
            "object_end_date": end if end is not None else "",
            "year_mid": year if year is not None else "",
            "year_source": year_source,
            "era": era,
            "era_confidence": era_confidence,
            "era_uncertain": "true" if is_uncertain else "false",
            "era_uncertain_reasons": ";".join(reasons),
            "match_source": match_source,
        }
        for col in EMOTION_COLS:
            row[col] = emo.get(col, "")

        rows_out.append(row)
        if is_uncertain:
            uncertain_rows.append(row)
        era_counts[era] += 1
        source_counts[year_source] += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows_out)

    uncertain_fields = [
        "filename",
        "artist",
        "title",
        "object_id",
        "object_date",
        "object_begin_date",
        "object_end_date",
        "year_mid",
        "year_source",
        "era",
        "era_confidence",
        "era_uncertain_reasons",
        "match_source",
    ]
    args.uncertain_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.uncertain_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=uncertain_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(uncertain_rows)

    print(f"Wrote {len(rows_out)} rows → {args.output_csv}")
    print(f"Uncertain rows → {args.uncertain_csv} ({len(uncertain_rows)})")
    print("Era distribution:")
    for era in ERA_BUCKETS:
        print(f"  {era}: {era_counts.get(era, 0)}")
    print("Year sources:")
    for src, n in sorted(source_counts.items()):
        print(f"  {src}: {n}")
    print(f"Uncertain rows: {uncertain_count}")
    print(f"Heuristic fallback rows: {heuristic_fallback_count}")

    if len(rows_out) != 104:
        print(f"WARNING: expected 104 rows, got {len(rows_out)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
