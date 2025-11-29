#!/usr/bin/env python3
"""
🍎 M1 최적화 셀카→초상화 변환기
Apple Silicon M1/M2에 최적화된 버전입니다.
"""

import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import mediapipe as mp

class M1OptimizedSelfieToPortrait:
    def __init__(self, lora_model_path="lora_trained_model/final"):
        """M1 최적화 셀카→초상화 변환기 초기화"""
        print("🍎 M1 최적화 셀카→초상화 변환기 초기화 중...")
        
        # M1/M2 GPU 사용 설정
        if torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
            print("🚀 Apple Silicon GPU (MPS) 사용!")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 NVIDIA GPU 사용!")
        else:
            self.device = "cpu"
            print("💻 CPU 사용")
        
        print(f"📱 사용 디바이스: {self.device}")
        
        # MediaPipe Face Detection 초기화
        print("👤 얼굴 검출 모델 로드 중...")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        print("✅ M1 최적화 변환기 초기화 완료!")
    
    def detect_face(self, image_path):
        """이미지에서 얼굴 검출"""
        print(f"👤 얼굴 검출 중: {image_path}")
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        # RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 검출
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            raise ValueError("얼굴을 찾을 수 없습니다!")
        
        print(f"✅ {len(results.detections)}개의 얼굴 발견!")
        return image, image_rgb, results.detections
    
    def analyze_emotion_from_image(self, image_path):
        """이미지에서 감정 분석 (간단한 방법)"""
        print("🔍 이미지 감정 분석 중...")
        
        try:
            # 얼굴 검출
            image, image_rgb, detections = self.detect_face(image_path)
            
            # 첫 번째 얼굴 분석
            face = detections[0]
            bbox = face.location_data.relative_bounding_box
            
            # 얼굴 영역 추출
            h, w = image_rgb.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            face_crop = image_rgb[y:y+height, x:x+width]
            
            # 간단한 색상 분석으로 감정 추정
            avg_color = np.mean(face_crop, axis=(0, 1))
            
            # 밝기 기반 감정 추정
            brightness = np.mean(avg_color)
            
            if brightness > 150:  # 밝은 이미지
                emotion = "joy"
            elif brightness < 100:  # 어두운 이미지
                emotion = "sadness"
            else:
                emotion = "neutral"
            
            print(f"💭 분석된 감정: {emotion} (밝기: {brightness:.1f})")
            return emotion, {emotion: 0.8}
            
        except Exception as e:
            print(f"⚠️ 감정 분석 실패: {e}")
            print("💡 기본값 'joy' 사용")
            return "joy", {"joy": 0.8}
    
    def get_emotion_prompt(self, emotion):
        """감정에 따른 프롬프트 생성"""
        emotion_prompts = {
            'joy': "a joyful portrait, happy expression, warm smile, bright eyes, cheerful, beautiful lighting, vibrant colors, masterpiece",
            'sadness': "a melancholic portrait, sad expression, contemplative gaze, gentle mood, soft lighting, muted colors, artistic",
            'anger': "a powerful portrait, intense expression, strong features, dramatic lighting, bold colors, dynamic",
            'fear': "a thoughtful portrait, cautious expression, mysterious atmosphere, subtle lighting, cool tones, introspective",
            'surprise': "an expressive portrait, surprised expression, wide eyes, dynamic pose, bright lighting, energetic",
            'disgust': "a dignified portrait, composed expression, refined features, elegant lighting, sophisticated, classical",
            'neutral': "a serene portrait, calm expression, peaceful atmosphere, balanced lighting, natural colors, tranquil",
            'love': "a tender portrait, gentle expression, warm atmosphere, caring eyes, romantic lighting, soft colors, intimate"
        }
        return emotion_prompts.get(emotion, emotion_prompts['joy'])
    
    def create_portrait_m1(self, image_path, output_path="my_portrait_m1.png"):
        """
        M1 최적화 초상화 생성
        """
        print(f"\n🎨 M1 최적화 초상화 생성 시작!")
        print(f"📸 입력 이미지: {image_path}")
        
        # 1. 감정 분석
        emotion, emotion_scores = self.analyze_emotion_from_image(image_path)
        
        # 2. 프롬프트 생성
        prompt = self.get_emotion_prompt(emotion)
        print(f"📝 생성 프롬프트: {prompt}")
        
        # 3. M1 최적화 파이프라인 로드
        print("🎨 M1 최적화 파이프라인 로드 중...")
        
        # 파이프라인 로드 (M1 최적화)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device in ["mps", "cuda"] else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # LoRA 모델 로드
        lora_path = "lora_trained_model/final"
        if os.path.exists(lora_path):
            print(f"🎭 LoRA 모델 로드: {lora_path}")
            pipe.load_lora_weights(lora_path)
        
        # M1 GPU로 이동
        if self.device == "mps":
            pipe = pipe.to(self.device)
            print("🚀 Apple Silicon GPU로 이동 완료!")
        elif self.device == "cuda":
            pipe = pipe.to(self.device)
            print("🚀 NVIDIA GPU로 이동 완료!")
        
        # 원본 이미지 로드 및 전처리
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        
        # 이미지 생성 (M1 최적화 설정)
        print("🖼️ 이미지 생성 중... (M1 GPU 가속 사용)")
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.7,  # 원본 얼굴 구조 유지
            guidance_scale=7.5,
            num_inference_steps=20,  # 빠른 생성
            generator=torch.Generator(device=self.device)  # M1 최적화
        )
        
        # 저장
        result.images[0].save(output_path)
        print(f"✅ 초상화 저장 완료: {output_path}")
        
        return result.images[0], emotion, emotion_scores


def main():
    parser = argparse.ArgumentParser(description="M1 최적화 셀카→초상화 변환")
    parser.add_argument("--input", type=str, required=True, help="입력 셀카 이미지 경로")
    parser.add_argument("--output", type=str, default="my_portrait_m1.png", help="출력 파일명")
    parser.add_argument("--lora-path", type=str, default="lora_trained_model/final", 
                       help="LoRA 모델 경로")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 변환기 초기화
    converter = M1OptimizedSelfieToPortrait(lora_model_path=args.lora_path)
    
    # 초상화 생성
    image, emotion, scores = converter.create_portrait_m1(
        args.input, 
        args.output
    )
    
    print(f"\n🎉 완료!")
    print(f"🎭 감정: {emotion}")
    print(f"📊 감정 점수: {scores}")
    print(f"💾 저장 위치: {args.output}")

if __name__ == "__main__":
    main()





