#!/usr/bin/env python3
"""
🎨 최종 셀카 → 포트레이트 변환 스크립트
사용자가 마음에 들어하는 기본 프롬프트를 사용합니다.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import mediapipe as mp

class SelfieToPortrait:
    def __init__(self):
        print("🎨 셀카 → 포트레이트 변환기 초기화 중...")
        
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
        
        print("✅ 초기화 완료!")
    
    def detect_face(self, image):
        """얼굴 감지"""
        # PIL을 OpenCV 형식으로 변환
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 얼굴 감지
        results = self.face_detection.process(cv_image)
        
        if results.detections:
            # 첫 번째 얼굴의 바운딩 박스 가져오기
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = cv_image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # 얼굴 영역 크롭
            face_crop = image.crop((x, y, x + width, y + height))
            
            print(f"✅ 얼굴 감지 성공: {width}x{height}")
            return face_crop, True
        else:
            print("❌ 얼굴을 찾을 수 없습니다.")
            return image, False
    
    def analyze_emotion_simple(self, face_image):
        """간단한 감정 분석"""
        # 얼굴 랜드마크 감지 (더 관대한 설정)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # 임계값 낮춤
            min_tracking_confidence=0.3
        )
        
        # PIL을 numpy로 변환
        img_array = np.array(face_image)
        
        # 얼굴 랜드마크 감지
        results = face_mesh.process(img_array)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # 입꼬리 좌표 (인덱스 61, 291)
            left_corner = landmarks.landmark[61]
            right_corner = landmarks.landmark[291]
            
            # 입꼬리 높이 차이 계산
            mouth_curve = right_corner.y - left_corner.y
            
            print(f"🔍 입꼬리 곡선 값: {mouth_curve:.4f}")
            
            # 간단한 감정 판단 (임계값 조정)
            if mouth_curve < -0.005:  # 입꼬리가 위로 올라감 (임계값 낮춤)
                emotion = "joyful"
                print("😊 감정 분석: 기쁨 (웃는 표정)")
            else:
                emotion = "neutral"
                print("😐 감정 분석: 중립")
        else:
            # 랜드마크 감지 실패 시 이미지 밝기로 감정 추정
            img_array = np.array(face_image)
            brightness = np.mean(img_array)
            
            if brightness > 120:  # 밝은 이미지
                emotion = "joyful"
                print("😊 감정 분석: 기쁨 (밝은 이미지로 추정)")
            else:
                emotion = "neutral"
                print("😐 감정 분석: 중립 (랜드마크 감지 실패, 밝기 기반)")
        
        return emotion
    
    def create_portrait(self, selfie_path):
        """셀카를 포트레이트로 변환"""
        print(f"\n🎨 셀카 처리 시작: {selfie_path}")
        
        # 이미지 로드
        try:
            image = Image.open(selfie_path).convert("RGB")
            print(f"✅ 이미지 로드 성공: {image.size}")
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {e}")
            return None
        
        # 얼굴 감지
        face_crop, face_found = self.detect_face(image)
        
        # 이미지 크기 조정 (512x512)
        input_image = face_crop.resize((512, 512))
        
        # 감정 분석
        emotion = self.analyze_emotion_simple(input_image)
        
        # 프롬프트 생성 (사용자가 마음에 들어하는 기본 프롬프트 사용)
        prompt = f"portrait painting, {emotion} expression"
        print(f"📝 사용 프롬프트: {prompt}")
        
        # 이미지 생성
        print("🎨 이미지 생성 중... (약 1-2분 소요)")
        try:
            result = self.pipe(
                prompt=prompt,
                image=input_image,
                strength=0.7,  # 원본 이미지와의 유사도
                guidance_scale=7.5,  # 프롬프트 준수도
                num_inference_steps=20,  # 품질을 위해 증가
                generator=torch.Generator(device=self.device)
            )
            
            # 결과 저장
            output_path = f"final_portrait_{emotion}.png"
            result.images[0].save(output_path)
            print(f"✅ 포트레이트 생성 완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 이미지 생성 실패: {e}")
            return None

def main():
    """메인 함수"""
    print("🎨 셀카 → 포트레이트 변환 시작!")
    
    # 변환기 초기화
    converter = SelfieToPortrait()
    
    # 셀카 파일 경로
    selfie_path = "IMG_5241.JPG"
    
    # 포트레이트 생성
    result_path = converter.create_portrait(selfie_path)
    
    if result_path:
        print(f"\n🎉 완료! 결과 이미지: {result_path}")
    else:
        print("\n❌ 변환 실패")

if __name__ == "__main__":
    main()
