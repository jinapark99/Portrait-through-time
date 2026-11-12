"""FastAPI app: health / match / generate + minimal static UI."""

from __future__ import annotations

import io
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Allow `uvicorn app.main:app` from repo root without installing a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.generation import (  # noqa: E402
    faceid_status,
    get_generator,
    list_era_loras,
    pick_device,
)
from data.era_emotion_matcher import (  # noqa: E402
    MODULE_VERSION,
    load_catalog,
    match_era_emotion,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portrait.app")

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Portrait Through Time",
    description="Selfie → emotion×era match → styled portrait (img2img MVP).",
    version="0.1.0",
)

# Catalog loaded once (cheap CSV).
_CATALOG = None


def get_catalog():
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog()
    return _CATALOG


def _read_upload_image(upload: UploadFile) -> Image.Image:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    return img


def _match_image(img: Image.Image) -> dict[str, Any]:
    # DeepFace accepts file paths most reliably; write a temp JPEG.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = Path(tmp.name)
        img.save(path, format="JPEG", quality=92)
    try:
        result = match_era_emotion(image=path, catalog=get_catalog(), top_k=3)
        return result.to_dict()
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/health")
def health() -> dict[str, Any]:
    faceid_ok, faceid_reason = faceid_status()
    gen = get_generator()
    return {
        "status": "ok",
        "service": "portrait-through-time",
        "matcher_version": MODULE_VERSION,
        "device": pick_device(),
        "catalog_size": len(get_catalog()),
        "generator_ready": gen.ready,
        "generator_load_error": gen.load_error,
        "lora_path": str(gen.lora_path) if gen.lora_path else None,
        "era_loras": list_era_loras(),
        "faceid_available": faceid_ok,
        "faceid_reason": faceid_reason,
        "outputs_dir": str(OUTPUTS_DIR),
    }


@app.post("/match")
async def match(file: UploadFile = File(...)) -> JSONResponse:
    img = _read_upload_image(file)
    try:
        payload = _match_image(img)
    except Exception as exc:
        logger.exception("match failed")
        raise HTTPException(status_code=500, detail=f"Match failed: {exc}") from exc
    return JSONResponse(payload)


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    strength: float = 0.72,
    steps: int = 24,
    guidance_scale: float = 7.0,
    save: bool = True,
) -> Response:
    """Match + generate. Returns PNG; match/meta in response headers (and JSON sibling via /generate/json)."""
    img = _read_upload_image(file)
    try:
        match_payload = _match_image(img)
    except Exception as exc:
        logger.exception("match-before-generate failed")
        raise HTTPException(status_code=500, detail=f"Match failed: {exc}") from exc

    prompt = match_payload.get("style_prompt") or "classical oil portrait painting"
    era = match_payload.get("era")
    gen = get_generator()
    try:
        result = gen.generate(
            img,
            prompt,
            era=str(era) if era else None,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
        )
    except Exception as exc:
        logger.exception("generate failed")
        raise HTTPException(status_code=503, detail=f"Generation failed: {exc}") from exc

    out_name = f"portrait_{uuid.uuid4().hex[:10]}.png"
    if save:
        out_path = OUTPUTS_DIR / out_name
        result.image.save(out_path)
        logger.info("saved %s", out_path)

    png = result.to_png_bytes()
    # HTTP headers must be latin-1; strip fancy punctuation from why-line.
    why_hdr = (
        str(match_payload.get("why_this_era", ""))
        .replace("—", "-")
        .replace("–", "-")
        .encode("ascii", "ignore")
        .decode("ascii")[:500]
    )
    headers = {
        "X-Portrait-Era": str(match_payload.get("era", "")),
        "X-Portrait-Emotion": str(
            (match_payload.get("emotion") or {}).get("dominant_emotion", "")
        ),
        "X-Portrait-Mode": result.meta.mode,
        "X-Portrait-Lora-Source": result.meta.lora_source,
        "X-Portrait-Why": why_hdr,
        "X-Portrait-Filename": out_name,
        "Content-Disposition": f'inline; filename="{out_name}"',
    }
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/generate/json")
async def generate_json(
    file: UploadFile = File(...),
    strength: float = 0.72,
    steps: int = 24,
    guidance_scale: float = 7.0,
) -> JSONResponse:
    """Match + generate; returns JSON with match, generation meta, and relative image URL."""
    import base64

    img = _read_upload_image(file)
    try:
        match_payload = _match_image(img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Match failed: {exc}") from exc

    prompt = match_payload.get("style_prompt") or "classical oil portrait painting"
    era = match_payload.get("era")
    gen = get_generator()
    try:
        result = gen.generate(
            img,
            prompt,
            era=str(era) if era else None,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Generation failed: {exc}") from exc

    out_name = f"portrait_{uuid.uuid4().hex[:10]}.png"
    out_path = OUTPUTS_DIR / out_name
    result.image.save(out_path)
    b64 = base64.b64encode(result.to_png_bytes()).decode("ascii")

    return JSONResponse(
        {
            "match": match_payload,
            "generation": result.meta.to_dict(),
            "image_filename": out_name,
            "image_path": str(out_path),
            "image_base64_png": b64,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>Portrait Through Time</h1><p>UI missing.</p>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
