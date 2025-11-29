#!/usr/bin/env python3
"""
🎨 감정별 회화 스타일 포트레이트 생성기
각 감정에 맞는 특정 회화 스타일을 적용합니다.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import mediapipe as mp

class EmotionStylePortrait:
    def __init__(self):
        print("🎨 감정별 회화 스타일 포트레이트 생성기 초기화 중...")
        
        # M1 GPU 설정
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"📱 사용 디바이스: {self.device}")
        
        # MediaPipe 얼굴 감지 초기화
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Stable Diffusion 파이프라인 로드
        print("🎨 Stable Diffusion 파이프라인 로드 중...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device in ["mps", "cuda"] else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)
        
        # LoRA 로드
        print("🎭 LoRA 모델 로드 중...")
        try:
            self.pipe.load_lora_weights("lora_trained_model/final")
            print("✅ LoRA 로드 성공!")
        except Exception as e:
            print(f"❌ LoRA 로드 실패: {e}")
            return
        
        # 감정별 회화 스타일 정의
        self.emotion_styles = {
            "joyful": {
                "style": "renaissance portrait, bright and warm colors, golden lighting, cheerful expression, baroque style, vibrant colors, optimistic mood",
                "description": "르네상스/바로크 스타일 - 밝고 따뜻한 색조"
            },
            "sad": {
                "style": "medieval portrait, dark and muted colors, melancholic expression, gothic style, somber mood, chiaroscuro lighting, contemplative",
                "description": "중세/고딕 스타일 - 어둡고 차분한 색조"
            },
            "angry": {
                "style": "baroque portrait, dramatic lighting, intense expression, bold colors, dynamic composition, powerful mood, dramatic shadows",
                "description": "바로크 스타일 - 드라마틱하고 강렬한 색조"
            },
            "surprised": {
                "style": "rococo portrait, elegant and refined, soft pastel colors, delicate expression, ornate details, graceful mood, refined style",
                "description": "로코코 스타일 - 우아하고 섬세한 색조"
            },
            "neutral": {
                "style": "classical portrait, balanced composition, natural colors, serene expression, timeless style, harmonious mood, classical proportions",
                "description": "클래식 스타일 - 균형잡힌 자연스러운 색조"
            },
            "contemplative": {
                "style": "romantic portrait, soft and dreamy colors, thoughtful expression, ethereal mood, soft lighting, introspective, romantic era style",
                "description": "낭만주의 스타일 - 부드럽고 몽환적인 색조"
            }
        }
        
        print("✅ 초기화 완료!")
    
    def detect_face(self, image):
        """얼굴 감지"""
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
            
            face_crop = image.crop((x, y, x + width, y + height))
            print(f"✅ 얼굴 감지 성공: {width}x{height}")
            return face_crop, True
        else:
            print("❌ 얼굴을 찾을 수 없습니다.")
            return image, False
    
    def analyze_emotion_advanced(self, face_image):
        """고급 감정 분석"""
        # 얼굴 랜드마크 감지
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
            
            # 다양한 얼굴 특징 분석
            # 입꼬리 (61, 291)
            left_corner = landmarks.landmark[61]
            right_corner = landmarks.landmark[291]
            mouth_curve = right_corner.y - left_corner.y
            
            # 눈썹 높이 (70, 300)
            left_eyebrow = landmarks.landmark[70]
            right_eyebrow = landmarks.landmark[300]
            eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
            
            # 눈 크기 (33, 7)
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[7]
            eye_openness = abs(left_eye.y - right_eye.y)
            
            print(f"🔍 얼굴 분석:")
            print(f"   입꼬리 곡선: {mouth_curve:.4f}")
            print(f"   눈썹 높이: {eyebrow_height:.4f}")
            print(f"   눈 크기: {eye_openness:.4f}")
            
            # 감정 판단 로직 (임계값 조정)
            if mouth_curve < -0.005 and eyebrow_height < 0.4:
                emotion = "joyful"
                print("😊 감정: 기쁨 (웃는 표정 + 낮은 눈썹)")
            elif mouth_curve > 0.01:  # 임계값 낮춤 (0.005 → 0.01)
                emotion = "sad"
                print("😢 감정: 슬픔 (내린 입꼬리)")
            elif eye_openness > 0.02 and eyebrow_height < 0.4:
                emotion = "surprised"
                print("😲 감정: 놀람 (크게 뜬 눈 + 낮은 눈썹)")
            elif eyebrow_height > 0.45 and mouth_curve < -0.002:
                emotion = "angry"
                print("😠 감정: 화남 (높은 눈썹 + 약간 웃는 입)")
            elif eyebrow_height > 0.42:
                emotion = "contemplative"
                print("🤔 감정: 사색 (높은 눈썹)")
            else:
                emotion = "neutral"
                print("😐 감정: 중립")
        else:
            # 랜드마크 감지 실패 시 더 보수적으로 판단
            img_array = np.array(face_image)
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            print(f"🔍 이미지 분석:")
            print(f"   밝기: {brightness:.2f}")
            print(f"   대비: {contrast:.2f}")
            
            # 더 보수적인 감정 판단 (랜드마크 없이는 중립으로)
            if brightness < 80:  # 매우 어두운 경우만 슬픔
                emotion = "sad"
                print("😢 감정: 슬픔 (매우 어두운 이미지)")
            elif brightness > 150 and contrast > 60:  # 매우 밝고 대비가 높은 경우만 기쁨
                emotion = "joyful"
                print("😊 감정: 기쁨 (매우 밝고 대비 높은 이미지)")
            else:
                emotion = "neutral"  # 기본값을 중립으로
                print("😐 감정: 중립 (랜드마크 감지 실패, 보수적 판단)")
        
        return emotion
    
    def get_emotion_style_prompt(self, emotion):
        """감정에 맞는 회화 스타일 프롬프트 반환"""
        if emotion in self.emotion_styles:
            style_info = self.emotion_styles[emotion]
            print(f"🎨 적용 스타일: {style_info['description']}")
            return style_info["style"]
        else:
            print(f"⚠️ 알 수 없는 감정: {emotion}, 기본 스타일 사용")
            return self.emotion_styles["neutral"]["style"]
    
    def create_emotion_portrait(self, selfie_path):
        """감정별 회화 스타일로 포트레이트 생성"""
        print(f"\n🎨 감정별 회화 스타일 포트레이트 생성 시작: {selfie_path}")
        
        # 이미지 로드
        try:
            image = Image.open(selfie_path).convert("RGB")
            print(f"✅ 이미지 로드 성공: {image.size}")
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {e}")
            return None
        
        # 얼굴 감지
        face_crop, face_found = self.detect_face(image)
        input_image = face_crop.resize((512, 512))
        
        # 고급 감정 분석
        emotion = self.analyze_emotion_advanced(input_image)
        
        # 감정별 회화 스타일 프롬프트 생성
        style_prompt = self.get_emotion_style_prompt(emotion)
        
        # 최종 프롬프트 생성
        final_prompt = f"portrait painting, {style_prompt}"
        print(f"📝 최종 프롬프트: {final_prompt}")
        
        # 이미지 생성
        print("🎨 감정별 회화 스타일 이미지 생성 중... (약 2-3분 소요)")
        try:
            result = self.pipe(
                prompt=final_prompt,
                image=input_image,
                strength=0.6,  # 스타일 적용을 위해 조정
                guidance_scale=8.0,  # 스타일 준수도 높임
                num_inference_steps=25,  # 품질 향상
                generator=torch.Generator(device=self.device)
            )
            
            # 결과 저장
            output_path = f"emotion_style_portrait_{emotion}.png"
            result.images[0].save(output_path)
            print(f"✅ 감정별 회화 스타일 포트레이트 생성 완료: {output_path}")
            
            return output_path, emotion
            
        except Exception as e:
            print(f"❌ 이미지 생성 실패: {e}")
            return None, None

def main():
    """메인 함수"""
    print("🎨 감정별 회화 스타일 포트레이트 생성 시작!")
    
    # 생성기 초기화
    generator = EmotionStylePortrait()
    
    # 셀카 파일 경로 (웹에서 다운로드한 테스트 이미지)
    selfie_path = "test_sad_face.jpg"
    
    # 감정별 회화 스타일 포트레이트 생성
    result_path, detected_emotion = generator.create_emotion_portrait(selfie_path)
    
    if result_path:
        print(f"\n🎉 완료!")
        print(f"   감지된 감정: {detected_emotion}")
        print(f"   결과 이미지: {result_path}")
        print(f"   적용된 스타일: {generator.emotion_styles[detected_emotion]['description']}")
    else:
        print("\n❌ 생성 실패")

if __name__ == "__main__":
    main()
