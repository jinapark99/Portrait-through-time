#!/usr/bin/env python3
"""
🎨 Premium Portrait Style Generator
A stunning web app with beautiful design that transforms your selfie into artistic portraits.
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import mediapipe as mp
import io
import time

# Page configuration
st.set_page_config(
    page_title="🎨 Portrait Style Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
    
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0;
        font-family: 'Inter', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 0 0 30px 30px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.8rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 2rem;
        font-weight: 300;
        font-style: italic;
        position: relative;
        z-index: 1;
    }
    
    .upload-card {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .upload-title {
        color: #2C3E50;
        font-size: 2rem;
        margin-bottom: 2rem;
        font-weight: 600;
        text-align: center;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .emotion-badge {
        display: inline-block;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 30px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .style-description {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #667eea;
        font-size: 1.1rem;
        color: #2C3E50;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .download-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 1.2rem 2.5rem;
        border-radius: 30px;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .download-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .footer {
        text-align: center;
        color: #7F8C8D;
        margin-top: 4rem;
        font-size: 1rem;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px 20px 0 0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    
    .feature-desc {
        color: #7F8C8D;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Custom file uploader styling */
    .stFileUploader > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed rgba(255,255,255,0.3);
    }
    
    .stFileUploader > div > div > div > div:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

class EmotionStylePortraitWeb:
    def __init__(self):
        # MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Emotion-based art styles
        self.emotion_styles = {
            "joyful": {
                "style": "renaissance portrait, bright and warm colors, golden lighting, cheerful expression, baroque style, vibrant colors, optimistic mood",
                "description": "Renaissance/Baroque Style - Bright and warm colors with golden lighting",
                "emoji": "😊"
            },
            "sad": {
                "style": "medieval portrait, dark and muted colors, melancholic expression, gothic style, somber mood, chiaroscuro lighting, contemplative",
                "description": "Medieval/Gothic Style - Dark and muted colors with dramatic lighting",
                "emoji": "😢"
            },
            "angry": {
                "style": "baroque portrait, dramatic lighting, intense expression, bold colors, dynamic composition, powerful mood, dramatic shadows",
                "description": "Baroque Style - Dramatic and intense with bold colors",
                "emoji": "😠"
            },
            "surprised": {
                "style": "rococo portrait, elegant and refined, soft pastel colors, delicate expression, ornate details, graceful mood, refined style",
                "description": "Rococo Style - Elegant and refined with soft pastel colors",
                "emoji": "😲"
            },
            "neutral": {
                "style": "classical portrait, balanced composition, natural colors, serene expression, timeless style, harmonious mood, classical proportions",
                "description": "Classical Style - Balanced and harmonious with natural colors",
                "emoji": "😐"
            },
            "contemplative": {
                "style": "romantic portrait, soft and dreamy colors, thoughtful expression, ethereal mood, soft lighting, introspective, romantic era style",
                "description": "Romantic Style - Soft and dreamy with ethereal mood",
                "emoji": "🤔"
            }
        }
        
        self.pipe = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    @st.cache_resource
    def load_model(_self):
        """Load AI model (cached)"""
        with st.spinner("🎨 Loading AI models... (This may take a moment on first run)"):
            # Load Stable Diffusion pipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if _self.device in ["mps", "cuda"] else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(_self.device)
            
            # Load LoRA
            try:
                pipe.load_lora_weights("lora_trained_model/final")
                st.success("✅ AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load LoRA: {e}")
                return None
        
        return pipe
    
    def detect_face(self, image):
        """Face detection with natural cropping"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.face_detection.process(cv_image)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = cv_image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Calculate face center
            face_center_x = x + width // 2
            face_center_y = y + height // 2
            
            # Natural cropping - expand 2.5x from face size
            crop_size = max(width, height) * 2.5
            
            # Calculate crop area (within image bounds)
            crop_x = max(0, int(face_center_x - crop_size // 2))
            crop_y = max(0, int(face_center_y - crop_size // 2))
            crop_x2 = min(w, int(face_center_x + crop_size // 2))
            crop_y2 = min(h, int(face_center_y + crop_size // 2))
            
            # Natural crop
            natural_crop = image.crop((crop_x, crop_y, crop_x2, crop_y2))
            return natural_crop, True
        else:
            return image, False
    
    def analyze_emotion_advanced(self, face_image):
        """Advanced emotion analysis"""
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        img_array = np.array(face_image)
        results = face_mesh.process(img_array)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Analyze facial features
            left_corner = landmarks.landmark[61]
            right_corner = landmarks.landmark[291]
            mouth_curve = right_corner.y - left_corner.y
            
            left_eyebrow = landmarks.landmark[70]
            right_eyebrow = landmarks.landmark[300]
            eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
            
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[7]
            eye_openness = abs(left_eye.y - right_eye.y)
            
            # Emotion detection logic
            if mouth_curve < -0.005 and eyebrow_height < 0.4:
                emotion = "joyful"
            elif mouth_curve > 0.01:
                emotion = "sad"
            elif eye_openness > 0.02 and eyebrow_height < 0.4:
                emotion = "surprised"
            elif eyebrow_height > 0.45 and mouth_curve < -0.002:
                emotion = "angry"
            elif eyebrow_height > 0.42:
                emotion = "contemplative"
            else:
                emotion = "neutral"
        else:
            # Fallback to image characteristics
            img_array = np.array(face_image)
            brightness = np.mean(img_array)
            
            if brightness < 80:
                emotion = "sad"
            elif brightness > 150:
                emotion = "joyful"
            else:
                emotion = "neutral"
        
        return emotion
    
    def create_portrait(self, image, emotion):
        """Create artistic portrait"""
        if self.pipe is None:
            self.pipe = self.load_model()
        
        if self.pipe is None:
            return None
        
        # Face detection
        face_crop, face_found = self.detect_face(image)
        input_image = face_crop.resize((512, 512))
        
        # Emotion-based style prompt
        style_prompt = self.emotion_styles[emotion]["style"]
        final_prompt = f"portrait painting, {style_prompt}"
        
        # Generate image with natural style transfer
        result = self.pipe(
            prompt=final_prompt,
            image=input_image,
            strength=0.4,  # Preserve original structure
            guidance_scale=7.5,  # Balanced style application
            num_inference_steps=20,
            generator=torch.Generator(device=self.device)
        )
        
        return result.images[0]

def main():
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Hero section
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">🎨 Portrait Style Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">If I were a painter, my self-portrait would be...</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Smart Upload</div>
            <div class="feature-desc">Simply upload your selfie and let AI do the magic</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎭</div>
            <div class="feature-title">Emotion Analysis</div>
            <div class="feature-desc">Advanced AI analyzes your facial expressions</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">Art Style</div>
            <div class="feature-desc">Each emotion gets its own unique painting style</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Processing</div>
            <div class="feature-desc">High-quality results in just a few minutes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="upload-title">📸 Upload Your Selfie</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear selfie with your face visible",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Your Selfie", use_container_width=True)
        
        # Generate button
        if st.button("🎨 Create My Portrait", type="primary", use_container_width=True):
            with st.spinner("🎨 Analyzing your emotion and creating your artistic portrait..."):
                # Initialize portrait generator
                generator = EmotionStylePortraitWeb()
                
                # Analyze emotion
                emotion = generator.analyze_emotion_advanced(image)
                
                # Create portrait
                portrait = generator.create_portrait(image, emotion)
                
                if portrait:
                    with col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        # Display emotion and style
                        emotion_info = generator.emotion_styles[emotion]
                        st.markdown(f"""
                        <div class="emotion-badge">
                            {emotion_info['emoji']} Detected Emotion: {emotion.title()}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="style-description">
                            <strong>Art Style Applied:</strong><br>
                            {emotion_info['description']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display portrait
                        st.image(portrait, caption="Your Artistic Portrait", use_container_width=True)
                        
                        # Download button
                        img_buffer = io.BytesIO()
                        portrait.save(img_buffer, format="PNG")
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Portrait",
                            data=img_buffer.getvalue(),
                            file_name=f"portrait_{emotion}_{int(time.time())}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to create portrait. Please try again.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎨 AI-Powered Emotion-Based Portrait Style Generator</p>
        <p>Powered by Stable Diffusion + LoRA + MediaPipe</p>
        <p>Transform your selfie into a masterpiece!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()





