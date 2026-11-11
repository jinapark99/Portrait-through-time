"""Era × emotion matcher for Portrait Through Time (Phase C / M3).

FastAPI-friendly pure functions: selfie path or image → DeepFace emotion →
1–3 reference portraits + era + style prompt + one-line why-this-era.

No Streamlit dependency. Deterministic ranking (no randomness).
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

EMOTION_KEYS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

# Plan fallback: Classical / Renaissance when era is empty after matching.
DEFAULT_ERA = "Renaissance"
DEFAULT_DETECTOR = "opencv"

ERA_STYLE_PROMPTS: dict[str, str] = {
    "Medieval/Gothic": (
        "medieval gothic panel portrait, gold ground, iconic frontal pose, "
        "tempera, flat decorative color, solemn sacred mood"
    ),
    "Renaissance": (
        "renaissance oil portrait, sfumato, balanced classical composition, "
        "soft natural light, lifelike flesh tones, serene dignity"
    ),
    "Baroque": (
        "baroque portrait, dramatic chiaroscuro, rich deep colors, "
        "dynamic composition, intense presence, tenebrism"
    ),
    "Rococo": (
        "rococo portrait, pastel palette, delicate brushwork, elegant ornament, "
        "soft diffused light, graceful refinement"
    ),
    "Romantic": (
        "romantic era portrait, soft dreamy atmosphere, emotional depth, "
        "warm muted tones, painterly brushwork, introspective mood"
    ),
    "Modern/Impressionist": (
        "impressionist modern portrait, visible brushstrokes, luminous color, "
        "contemporary informal pose, outdoor light"
    ),
}

EMOTION_EXPRESSION_CUES: dict[str, str] = {
    "angry": "intense expression, furrowed brow, strong features",
    "disgust": "composed disdain, tightened mouth, averted gaze",
    "fear": "wary eyes, tense expression, uneasy atmosphere",
    "happy": "warm smile, bright eyes, cheerful expression",
    "sad": "melancholic expression, downcast gaze, contemplative sorrow",
    "surprise": "wide eyes, raised brows, open expressive face",
    "neutral": "calm expression, balanced features, quiet presence",
}

ERA_WHY_TEMPLATES: dict[str, str] = {
    "Medieval/Gothic": (
        "Your {emotion} expression maps to Gothic panel faces — solemn, iconic, and frontal."
    ),
    "Renaissance": (
        "Your {emotion} expression aligns with Classical/Renaissance portrait balance and sfumato calm."
    ),
    "Baroque": (
        "Your {emotion} expression fits Baroque drama — chiaroscuro and psychologically charged presence."
    ),
    "Rococo": (
        "Your {emotion} expression echoes Rococo elegance — soft pastels and refined, intimate posing."
    ),
    "Romantic": (
        "Your {emotion} mood matches Romantic portraiture — inward feeling over courtly display."
    ),
    "Modern/Impressionist": (
        "Your {emotion} reading pairs with Modern/Impressionist portraits — informal light and living color."
    ),
}

# When voting among near-ties, prefer historically "classical" eras first.
ERA_TIEBREAK_ORDER: tuple[str, ...] = (
    "Renaissance",
    "Baroque",
    "Romantic",
    "Rococo",
    "Modern/Impressionist",
    "Medieval/Gothic",
)

MATCH_POOL = 15
MODULE_VERSION = "1.0.0"


@dataclass(frozen=True)
class EmotionAnalysis:
    dominant_emotion: str
    scores: dict[str, float]
    top_emotions: list[tuple[str, float]]
    face_detected: bool
    confidence: float
    detector: str
    source: str = "deepface"  # deepface | scores | dominant

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_emotion": self.dominant_emotion,
            "scores": dict(self.scores),
            "top_emotions": [{"emotion": e, "score": s} for e, s in self.top_emotions],
            "face_detected": self.face_detected,
            "confidence": self.confidence,
            "detector": self.detector,
            "source": self.source,
        }


@dataclass(frozen=True)
class ReferencePortrait:
    filename: str
    path: str
    era: str
    dominant_emotion: str
    artist: str
    title: str
    similarity: float
    face_detected: bool
    emotion_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class MatchResult:
    emotion: EmotionAnalysis
    era: str
    match_mode: str  # era_emotion | emotion_only_fallback | default_era
    references: list[ReferencePortrait]
    style_prompt: str
    why_this_era: str
    catalog_size: int
    module_version: str = MODULE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion.to_dict(),
            "era": self.era,
            "match_mode": self.match_mode,
            "references": [r.to_dict() for r in self.references],
            "style_prompt": self.style_prompt,
            "why_this_era": self.why_this_era,
            "catalog_size": self.catalog_size,
            "module_version": self.module_version,
        }


@dataclass
class CatalogRow:
    filename: str
    artist: str
    title: str
    era: str
    dominant_emotion: str
    face_detected: bool
    confidence: float
    scores: dict[str, float]
    path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_labels_csv() -> Path:
    return repo_root() / "data" / "labels" / "portraits_104_era_emotion.csv"


def default_images_dir() -> Path:
    return repo_root() / "data" / "lora_training_data" / "images"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_emotion_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    """Return a full 7-key emotion vector (missing keys → 0)."""
    out = {k: max(0.0, _as_float(scores.get(k, 0.0))) for k in EMOTION_KEYS}
    total = sum(out.values())
    # DeepFace often returns 0–100 percentages; keep raw scale for cosine.
    if total <= 0:
        out["neutral"] = 100.0
    return out


def dominant_from_scores(scores: Mapping[str, float]) -> str:
    norm = normalize_emotion_scores(scores)
    return max(EMOTION_KEYS, key=lambda k: (norm[k], -EMOTION_KEYS.index(k)))


def top_emotions(scores: Mapping[str, float], k: int = 2) -> list[tuple[str, float]]:
    norm = normalize_emotion_scores(scores)
    ranked = sorted(norm.items(), key=lambda kv: (-kv[1], EMOTION_KEYS.index(kv[0])))
    return ranked[: max(1, k)]


def cosine_similarity(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    va = normalize_emotion_scores(a)
    vb = normalize_emotion_scores(b)
    dot = sum(va[k] * vb[k] for k in EMOTION_KEYS)
    na = math.sqrt(sum(va[k] * va[k] for k in EMOTION_KEYS))
    nb = math.sqrt(sum(vb[k] * vb[k] for k in EMOTION_KEYS))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def load_catalog(
    labels_csv: Path | str | None = None,
    images_dir: Path | str | None = None,
) -> list[CatalogRow]:
    csv_path = Path(labels_csv) if labels_csv else default_labels_csv()
    img_dir = Path(images_dir) if images_dir else default_images_dir()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    rows: list[CatalogRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            filename = (raw.get("filename") or "").strip()
            if not filename:
                continue
            era = (raw.get("era") or "").strip()
            scores = {k: _as_float(raw.get(k)) for k in EMOTION_KEYS}
            dominant = (raw.get("dominant_emotion") or "").strip().lower()
            if not dominant or dominant == "unlabeled":
                dominant = dominant_from_scores(scores)
            rows.append(
                CatalogRow(
                    filename=filename,
                    artist=(raw.get("artist") or "").strip(),
                    title=(raw.get("title") or "").strip(),
                    era=era,
                    dominant_emotion=dominant,
                    face_detected=_as_bool(raw.get("face_detected")),
                    confidence=_as_float(raw.get("confidence")),
                    scores=normalize_emotion_scores(scores),
                    path=img_dir / filename,
                )
            )
    # Deterministic order for any later stable sorts.
    rows.sort(key=lambda r: r.filename)
    return rows


def analyze_selfie_emotion(
    image: str | Path | Any,
    *,
    detector_backend: str = DEFAULT_DETECTOR,
    enforce_detection: bool = False,
) -> EmotionAnalysis:
    """Run DeepFace emotion analysis on a path, ndarray, or PIL image."""
    from deepface import DeepFace

    img_arg: Any
    if isinstance(image, (str, Path)):
        img_arg = str(image)
    else:
        img_arg = image

    result = DeepFace.analyze(
        img_path=img_arg,
        actions=["emotion"],
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        silent=True,
    )
    if isinstance(result, list):
        result = result[0]

    raw_scores = result.get("emotion") or {}
    scores = normalize_emotion_scores(raw_scores)
    dominant = str(result.get("dominant_emotion") or dominant_from_scores(scores)).lower()
    if dominant not in EMOTION_KEYS:
        dominant = dominant_from_scores(scores)

    conf = _as_float(result.get("face_confidence"), default=0.0)
    face_detected = conf > 0.0
    return EmotionAnalysis(
        dominant_emotion=dominant,
        scores=scores,
        top_emotions=top_emotions(scores, k=2),
        face_detected=face_detected,
        confidence=conf,
        detector=detector_backend,
        source="deepface",
    )


def emotion_from_scores(
    scores: Mapping[str, Any],
    *,
    dominant: str | None = None,
    face_detected: bool = True,
    confidence: float = 1.0,
    source: str = "scores",
) -> EmotionAnalysis:
    norm = normalize_emotion_scores(scores)
    dom = (dominant or dominant_from_scores(norm)).lower()
    if dom not in EMOTION_KEYS:
        dom = dominant_from_scores(norm)
    return EmotionAnalysis(
        dominant_emotion=dom,
        scores=norm,
        top_emotions=top_emotions(norm, k=2),
        face_detected=face_detected,
        confidence=confidence,
        detector="none",
        source=source,
    )


def build_style_prompt(era: str, dominant_emotion: str) -> str:
    era_prompt = ERA_STYLE_PROMPTS.get(era) or ERA_STYLE_PROMPTS[DEFAULT_ERA]
    cue = EMOTION_EXPRESSION_CUES.get(dominant_emotion, EMOTION_EXPRESSION_CUES["neutral"])
    return f"portrait painting, {era_prompt}, {cue}"


def why_this_era(era: str, dominant_emotion: str) -> str:
    template = ERA_WHY_TEMPLATES.get(era) or ERA_WHY_TEMPLATES[DEFAULT_ERA]
    return template.format(emotion=dominant_emotion)


def _era_ok(era: str) -> bool:
    return bool(era) and era.strip().lower() not in {"", "unlabeled", "unknown", "none"}


def _rank_candidates(
    catalog: Sequence[CatalogRow],
    query_scores: Mapping[str, float],
) -> list[tuple[CatalogRow, float]]:
    ranked: list[tuple[CatalogRow, float]] = []
    for row in catalog:
        sim = cosine_similarity(query_scores, row.scores)
        # Prefer face-detected rows without discarding others.
        face_boost = 0.001 if row.face_detected else 0.0
        ranked.append((row, sim + face_boost))
    ranked.sort(key=lambda item: (-item[1], item[0].filename))
    # Return raw cosine (without boost) for reporting.
    out: list[tuple[CatalogRow, float]] = []
    for row, _ in ranked:
        out.append((row, cosine_similarity(query_scores, row.scores)))
    return out


def _modal_era(scored: Sequence[tuple[CatalogRow, float]]) -> str | None:
    if not scored:
        return None
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row, sim in scored:
        if not _era_ok(row.era):
            continue
        totals[row.era] = totals.get(row.era, 0.0) + sim
        counts[row.era] = counts.get(row.era, 0) + 1
    if not counts:
        return None

    def sort_key(era: str) -> tuple:
        tie = ERA_TIEBREAK_ORDER.index(era) if era in ERA_TIEBREAK_ORDER else 99
        return (-counts[era], -totals[era], tie, era)

    return sorted(counts.keys(), key=sort_key)[0]


def _to_reference(row: CatalogRow, similarity: float) -> ReferencePortrait:
    return ReferencePortrait(
        filename=row.filename,
        path=str(row.path),
        era=row.era if _era_ok(row.era) else DEFAULT_ERA,
        dominant_emotion=row.dominant_emotion,
        artist=row.artist,
        title=row.title,
        similarity=round(similarity, 6),
        face_detected=row.face_detected,
        emotion_scores=dict(row.scores),
    )


def match_from_emotion(
    emotion: EmotionAnalysis,
    catalog: Sequence[CatalogRow] | None = None,
    *,
    top_k: int = 3,
    labels_csv: Path | str | None = None,
    images_dir: Path | str | None = None,
) -> MatchResult:
    """Pick era + 1–3 refs from a labeled catalog given an emotion analysis."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    rows = list(catalog) if catalog is not None else load_catalog(labels_csv, images_dir)
    query = emotion.scores
    dominant = emotion.dominant_emotion

    same_emotion = [r for r in rows if r.dominant_emotion == dominant and _era_ok(r.era)]
    ranked_same = _rank_candidates(same_emotion, query)

    match_mode = "era_emotion"
    chosen_era: str | None = None
    refs: list[ReferencePortrait] = []

    if ranked_same:
        pool = ranked_same[:MATCH_POOL]
        chosen_era = _modal_era(pool)
        if chosen_era:
            era_hits = [(r, s) for r, s in ranked_same if r.era == chosen_era]
            refs = [_to_reference(r, s) for r, s in era_hits[:top_k]]
            # If the era×emotion bucket is thin, fill from other same-emotion eras.
            if len(refs) < top_k:
                seen = {ref.filename for ref in refs}
                for r, s in ranked_same:
                    if r.filename in seen:
                        continue
                    refs.append(_to_reference(r, s))
                    if len(refs) >= top_k:
                        break
        else:
            match_mode = "emotion_only_fallback"
            refs = [_to_reference(r, s) for r, s in ranked_same[:top_k]]
            chosen_era = refs[0].era if refs else None

    if not refs:
        # Emotion-only: ignore dominant label, use vector similarity across catalog.
        match_mode = "emotion_only_fallback"
        any_era = [r for r in rows if _era_ok(r.era)]
        ranked_any = _rank_candidates(any_era or list(rows), query)
        if ranked_any:
            pool = ranked_any[:MATCH_POOL]
            chosen_era = _modal_era(pool) or (
                ranked_any[0][0].era if _era_ok(ranked_any[0][0].era) else None
            )
            if chosen_era and _era_ok(chosen_era):
                era_hits = [(r, s) for r, s in ranked_any if r.era == chosen_era]
                pick = era_hits[:top_k] if era_hits else ranked_any[:top_k]
            else:
                pick = ranked_any[:top_k]
            refs = [_to_reference(r, s) for r, s in pick]

    if not _era_ok(chosen_era or ""):
        match_mode = "default_era"
        chosen_era = DEFAULT_ERA
        if not refs:
            default_rows = [r for r in rows if r.era == DEFAULT_ERA]
            ranked_default = _rank_candidates(default_rows or list(rows), query)
            refs = [_to_reference(r, s) for r, s in ranked_default[:top_k]]

    assert chosen_era is not None
    return MatchResult(
        emotion=emotion,
        era=chosen_era,
        match_mode=match_mode,
        references=refs[:top_k],
        style_prompt=build_style_prompt(chosen_era, dominant),
        why_this_era=why_this_era(chosen_era, dominant),
        catalog_size=len(rows),
    )


def match_era_emotion(
    image: str | Path | Any | None = None,
    *,
    scores: Mapping[str, Any] | None = None,
    dominant_emotion: str | None = None,
    catalog: Sequence[CatalogRow] | None = None,
    labels_csv: Path | str | None = None,
    images_dir: Path | str | None = None,
    top_k: int = 3,
    detector_backend: str = DEFAULT_DETECTOR,
    enforce_detection: bool = False,
) -> MatchResult:
    """API-ready entry: image and/or emotion scores → MatchResult.

    Prefer ``image`` (path / ndarray / PIL) for live DeepFace analysis.
    Pass ``scores`` / ``dominant_emotion`` to skip DeepFace (tests, fixtures).
    """
    if image is None and scores is None and not dominant_emotion:
        raise ValueError("Provide image, scores, or dominant_emotion")

    if scores is not None or (dominant_emotion and image is None):
        if scores is None:
            # One-hot-ish fixture for a named dominant emotion.
            scores = {k: (100.0 if k == dominant_emotion else 0.0) for k in EMOTION_KEYS}
        emotion = emotion_from_scores(
            scores,
            dominant=dominant_emotion,
            source="scores" if scores is not None else "dominant",
        )
    else:
        emotion = analyze_selfie_emotion(
            image,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )

    return match_from_emotion(
        emotion,
        catalog,
        top_k=top_k,
        labels_csv=labels_csv,
        images_dir=images_dir,
    )
