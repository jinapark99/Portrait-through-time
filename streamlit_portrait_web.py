#!/usr/bin/env python3
"""
🌐 감정별 회화 스타일 포트레이트 웹 서비스
Streamlit으로 만든 웹 앱 - 셀카 업로드 → 감정 분석 → 회화 스타일 포트레이트 생성
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

# 페이지 설정
st.set_page_config(
    page_title="🎨 감정별 회화 스타일 포트레이트",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    .processing-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class EmotionStylePortraitWeb:
    def __init__(self):
        # MediaPipe 얼굴 감지 초기화
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # 감정별 회화 스타일 정의
        self.emotion_styles = {
            "joyful": {
                "style": "renaissance portrait, bright and warm colors, golden lighting, cheerful expression, baroque style, vibrant colors, optimistic mood",
                "description": "르네상스/바로크 스타일 - 밝고 따뜻한 색조",
                "emoji": "😊"
            },
            "sad": {
                "style": "medieval portrait, dark and muted colors, melancholic expression, gothic style, somber mood, chiaroscuro lighting, contemplative",
                "description": "중세/고딕 스타일 - 어둡고 차분한 색조",
                "emoji": "😢"
            },
            "angry": {
                "style": "baroque portrait, dramatic lighting, intense expression, bold colors, dynamic composition, powerful mood, dramatic shadows",
                "description": "바로크 스타일 - 드라마틱하고 강렬한 색조",
                "emoji": "😠"
            },
            "surprised": {
                "style": "rococo portrait, elegant and refined, soft pastel colors, delicate expression, ornate details, graceful mood, refined style",
                "description": "로코코 스타일 - 우아하고 섬세한 색조",
                "emoji": "😲"
            },
            "neutral": {
                "style": "classical portrait, balanced composition, natural colors, serene expression, timeless style, harmonious mood, classical proportions",
                "description": "클래식 스타일 - 균형잡힌 자연스러운 색조",
                "emoji": "😐"
            },
            "contemplative": {
                "style": "romantic portrait, soft and dreamy colors, thoughtful expression, ethereal mood, soft lighting, introspective, romantic era style",
                "description": "낭만주의 스타일 - 부드럽고 몽환적인 색조",
                "emoji": "🤔"
            }
        }
        
        self.pipe = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    @st.cache_resource
    def load_model(_self):
        """모델 로드 (캐시됨)"""
        with st.spinner("🎨 AI 모델 로드 중... (처음 실행 시 시간이 걸릴 수 있습니다)"):
            # Stable Diffusion 파이프라인 로드
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if _self.device in ["mps", "cuda"] else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(_self.device)
            
            # LoRA 로드
            try:
                pipe.load_lora_weights("lora_trained_model/final")
                st.success("✅ AI 모델 로드 완료!")
            except Exception as e:
                st.error(f"❌ LoRA 로드 실패: {e}")
                return None
        
        return pipe
    
    def detect_face(self, image):
        """얼굴 감지 및 자연스러운 크롭"""
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
            
            # 얼굴 중심점 계산
            face_center_x = x + width // 2
            face_center_y = y + height // 2
            
            # 자연스러운 크롭을 위해 얼굴 크기의 2.5배로 확장
            crop_size = max(width, height) * 2.5
            
            # 크롭 영역 계산 (이미지 경계 내에서)
            crop_x = max(0, int(face_center_x - crop_size // 2))
            crop_y = max(0, int(face_center_y - crop_size // 2))
            crop_x2 = min(w, int(face_center_x + crop_size // 2))
            crop_y2 = min(h, int(face_center_y + crop_size // 2))
            
            # 자연스러운 크롭
            natural_crop = image.crop((crop_x, crop_y, crop_x2, crop_y2))
            return natural_crop, True
        else:
            return image, False
    
    def analyze_emotion_advanced(self, face_image):
        """고급 감정 분석"""
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
            
            # 얼굴 특징 분석
            left_corner = landmarks.landmark[61]
            right_corner = landmarks.landmark[291]
            mouth_curve = right_corner.y - left_corner.y
            
            left_eyebrow = landmarks.landmark[70]
            right_eyebrow = landmarks.landmark[300]
            eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
            
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[7]
            eye_openness = abs(left_eye.y - right_eye.y)
            
            # 감정 판단
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
            # 이미지 특성으로 판단
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
        """포트레이트 생성"""
        if self.pipe is None:
            self.pipe = self.load_model()
        
        if self.pipe is None:
            return None
        
        # 얼굴 감지
        face_crop, face_found = self.detect_face(image)
        input_image = face_crop.resize((512, 512))
        
        # 감정별 스타일 프롬프트
        style_prompt = self.emotion_styles[emotion]["style"]
        final_prompt = f"portrait painting, {style_prompt}"
        
        # 이미지 생성 (더 자연스러운 결과를 위해 strength 조정)
        result = self.pipe(
            prompt=final_prompt,
            image=input_image,
            strength=0.4,  # 원본 이미지 구조를 더 보존
            guidance_scale=7.5,  # 프롬프트 준수도 조정
            num_inference_steps=20,
            generator=torch.Generator(device=self.device)
        )
        
        return result.images[0]

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🎨 감정별 회화 스타일 포트레이트</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">셀카를 업로드하면 AI가 감정을 분석해서 맞는 회화 스타일로 포트레이트를 생성합니다!</p>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("🎭 감정별 회화 스타일")
        
        for emotion, info in EmotionStylePortraitWeb().emotion_styles.items():
            st.markdown(f"**{info['emoji']} {emotion.title()}**")
            st.markdown(f"*{info['description']}*")
            st.markdown("---")
        
        st.markdown("""
        ### 📝 사용법
        1. 셀카 이미지 업로드
        2. AI가 감정 분석
        3. 맞는 회화 스타일로 포트레이트 생성
        4. 결과 다운로드
        """)
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 셀카 업로드")
        
        uploaded_file = st.file_uploader(
            "이미지를 선택하세요",
            type=['jpg', 'jpeg', 'png'],
            help="얼굴이 명확히 보이는 셀카를 업로드해주세요"
        )
        
        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="업로드된 이미지", use_column_width=True)
            
            # 처리 버튼
            if st.button("🎨 포트레이트 생성하기", type="primary"):
                with st.spinner("AI가 감정을 분석하고 포트레이트를 생성 중입니다..."):
                    # 포트레이트 생성기 초기화
                    generator = EmotionStylePortraitWeb()
                    
                    # 감정 분석
                    emotion = generator.analyze_emotion_advanced(image)
                    
                    # 포트레이트 생성
                    portrait = generator.create_portrait(image, emotion)
                    
                    if portrait:
                        with col2:
                            st.header("🎨 생성된 포트레이트")
                            
                            # 감정 정보 표시
                            emotion_info = generator.emotion_styles[emotion]
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>{emotion_info['emoji']} 감지된 감정: {emotion.title()}</h3>
                                <p><strong>적용된 스타일:</strong> {emotion_info['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 포트레이트 표시
                            st.image(portrait, caption="생성된 포트레이트", use_column_width=True)
                            
                            # 다운로드 버튼
                            img_buffer = io.BytesIO()
                            portrait.save(img_buffer, format="PNG")
                            img_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 포트레이트 다운로드",
                                data=img_buffer.getvalue(),
                                file_name=f"portrait_{emotion}_{int(time.time())}.png",
                                mime="image/png"
                            )
                    else:
                        st.error("❌ 포트레이트 생성에 실패했습니다.")
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎨 AI 감정 분석 + 회화 스타일 포트레이트 생성 서비스</p>
        <p>Powered by Stable Diffusion + LoRA + MediaPipe</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
