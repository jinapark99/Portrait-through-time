# 🎨 Portrait Through Time
*"How would classical artists paint me?"*

A machine learning project that transforms selfies into artistic portraits using emotion detection and classical art style mapping. This project explores how historical artists from different eras would interpret modern faces and expressions.

## 🌟 Features

- **📸 Selfie Upload** - Drag & drop or file selection
- **🎭 Emotion Analysis** - AI-powered facial emotion detection
- **🎨 Art Style Mapping** - Different painting styles for each emotion
- **⚡ Fast Processing** - M1 GPU acceleration
- **💾 Download** - High-quality PNG export

## 🎭 Emotion-Based Art Styles

- **😊 Joyful** → Renaissance/Baroque (Bright and warm colors)
- **😢 Sad** → Medieval/Gothic (Dark and muted colors)
- **😠 Angry** → Baroque (Dramatic and intense)
- **😲 Surprised** → Rococo (Elegant and refined)
- **🤔 Contemplative** → Romantic (Soft and dreamy)
- **😐 Neutral** → Classical (Balanced and harmonious)

## 🚀 Quick Start

1. Upload your selfie
2. Click "Create My Portrait"
3. AI analyzes your emotion
4. Get your artistic portrait!

## 🛠️ Technical Stack

- **Web MVP**: FastAPI + static HTML/JS (`app/`)
- **Matching**: DeepFace emotion → `data/era_emotion_matcher.py` (era × emotion catalog)
- **Generation (D1)**: SD1.5 img2img + era prompt + **era LoRA when available** (else legacy mixed LoRA / prompt-only); IP-Adapter FaceID stubbed unless extras are installed
- **Era LoRA**: `models/era_lora/<slug>/final` — Modern/Renaissance/Baroque/Romantic (300 steps each, MPS); Rococo/Medieval skipped. Demo defaults: img2img strength **0.82**, LoRA scale **1.6**
- **Legacy UI**: Streamlit (`streamlit_portrait_web.py`) — local fallback only
- **GPU**: Apple Silicon MPS / CUDA / CPU

## Web MVP — run locally

```bash
cd /path/to/Portrait
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# From repo root:
uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000
```

### API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Device, catalog size, FaceID / LoRA status |
| `POST` | `/match` | Upload selfie → JSON emotion + era + why + refs |
| `POST` | `/generate` | Upload selfie → PNG (headers carry era/emotion) |
| `POST` | `/generate/json` | Match + generate → JSON with base64 PNG + meta |

### Smoke test (no server)

```bash
python scripts/smoke_web_mvp.py --image test_happy_face.jpg --steps 16
# Match only:
python scripts/smoke_web_mvp.py --match-only --image test_sad_face.jpg
# Era LoRA before/after (needs models/era_lora/.../final):
python scripts/smoke_era_lora.py --era "Modern/Impressionist" --image test_happy_face.jpg
```

Train / continue era adapters:

```bash
python scripts/split_era_lora_datasets.py
python scripts/train_lora_era.py --era modern_impressionist --max-steps 80 --epochs 8
# Fuller Baroque/Romantic: see docs/era-lora-colab.md
```

Env knobs: `PORTRAIT_BASE_MODEL`, `PORTRAIT_LORA_PATH`, `PORTRAIT_DISABLE_FACEID=1`.

**What works today:** match UI + DeepFace; img2img on MPS with **era-matched LoRA** when `models/era_lora/<slug>/final` exists (Modern trained); otherwise legacy `lora_trained_model/final` or prompt-only. FaceID stubbed until `insightface` + IP-Adapter FaceID weights are installed.

**What’s left for public deploy:** GPU worker (or HF Space GPU), model bake into image / volume, FaceID wiring, and hosting secrets. Do not expect a free Railway/Fly CPU dyno to run SD1.5 comfortably.

## Suggested deploy

| Target | Fit | Notes |
|--------|-----|-------|
| **Hugging Face Space (Docker + GPU)** | Best single-box demo | Bake `requirements.txt` + cache SD1.5; enable GPU hardware |
| **Fly.io / Railway** | Web + match only on CPU | Put `/generate` behind a GPU worker (Modal / Replicate / RunPod) later |
| **Modal / Replicate** | Generation worker | Call from FastAPI; keep web tier light |

LoRA weights under `lora_trained_model/` are gitignored (large). Mount them at deploy time or ship without LoRA (prompt-only img2img still runs).

## 📱 Live Demo

No public URL yet — run locally with uvicorn (above).

## 💭 Motivation

What would I look like if painted by Renaissance masters, Baroque artists, or Romantic painters? This curiosity drove the project.

I wanted to explore:
1. **Historical Perspective**: How would artists from different eras interpret my face and expression?
2. **Emotion-Style Connection**: Does the emotion I show influence which art style best captures it?
3. **Temporal Bridge**: Using AI to connect past artistic traditions with present self-expression.

The project combines classical portrait datasets with modern emotion detection to generate portraits that reflect both my current expression and historical painting styles. It's an experiment in seeing myself through the eyes of artists from different periods.

## 🎨 How It Works

1. **Face Detection** - MediaPipe detects facial features
2. **Emotion Analysis** - AI analyzes facial landmarks and expressions
3. **Style Mapping** - Maps emotions to specific art styles
4. **Portrait Generation** - Stable Diffusion + LoRA creates artistic portrait
5. **Download** - High-quality result ready for sharing

## 📚 Project Structure

- `app/main.py` — FastAPI health / match / generate + static UI
- `app/generation.py` — img2img (+ FaceID stub) generation module
- `data/era_emotion_matcher.py` — era × emotion matching
- `data/labels/portraits_104_era_emotion.csv` — labeled catalog
- `scripts/smoke_web_mvp.py` — CLI smoke test
- `streamlit_portrait_web.py` — legacy Streamlit UI
- `selfie_to_portrait.py` / `portrait.py` — older conversion / training scripts

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 🎓 Training

To train the LoRA model on portrait datasets:

```bash
python portrait.py
```

---

*"If I were a painter, my self-portrait would be..."*