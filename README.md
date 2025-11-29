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

- **Frontend**: Streamlit
- **AI Models**: Stable Diffusion + LoRA
- **Face Detection**: MediaPipe
- **Emotion Analysis**: Custom facial landmark analysis
- **GPU**: M1 Metal Performance Shaders

## 📱 Live Demo

Visit: [Your Streamlit Cloud URL]

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

- `portrait.py` - LoRA fine-tuning script for training on portrait datasets
- `streamlit_portrait_web.py` - Web application for portrait generation
- `download_images_fast.py` - Script for downloading portrait datasets from public museums
- `selfie_to_portrait.py` - Core conversion script

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