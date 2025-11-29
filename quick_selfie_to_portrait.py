#!/usr/bin/env python3
"""
🚀 빠른 셀카→초상화 변환기 (간단 버전)
DeepFace 없이 MediaPipe만 사용해서 빠르게 작동합니다.
"""

import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import mediapipe as mp

class QuickSelfieToPortrait:
    def __init__(self, lora_model_path="lora_trained_model/final"):
        """빠른 셀카→초상화 변환기 초기화"""
        print("🚀 빠른 셀카→초상화 변환기 초기화 중...")
        
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 사용 디바이스: {self.device}")
        
        # MediaPipe Face Detection 초기화
        print("👤 얼굴 검출 모델 로드 중...")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        print("✅ 빠른 변환기 초기화 완료!")
    
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
    
    def simple_emotion_analysis(self, image_path):
        """간단한 감정 분석 (기본값)"""
        print("💭 간단한 감정 분석: 기본값 'joy' 사용")
        return "joy", {"joy": 0.8}
    
    def get_emotion_prompt(self, emotion):
        """감정에 따른 프롬프트 생성"""
        emotion_prompts = {
            'joy': "a joyful portrait, happy expression, warm smile, bright eyes, cheerful, beautiful lighting",
            'sadness': "a melancholic portrait, sad expression, contemplative gaze, gentle mood, soft lighting",
            'anger': "a powerful portrait, intense expression, strong features, dramatic lighting",
            'fear': "a thoughtful portrait, cautious expression, mysterious atmosphere, subtle lighting",
            'surprise': "an expressive portrait, surprised expression, wide eyes, dynamic pose",
            'disgust': "a dignified portrait, composed expression, refined features, elegant lighting",
            'neutral': "a serene portrait, calm expression, peaceful atmosphere, balanced lighting",
            'love': "a tender portrait, gentle expression, warm atmosphere, caring eyes, romantic lighting"
        }
        return emotion_prompts.get(emotion, emotion_prompts['joy'])
    
    def create_portrait_fast(self, image_path, output_path="my_portrait_fast.png", emotion="joy"):
        """
        빠른 초상화 생성 (Img2Img 사용)
        """
        print(f"\n🎨 빠른 초상화 생성 시작!")
        print(f"📸 입력 이미지: {image_path}")
        print(f"🎭 사용할 감정: {emotion}")
        
        # 1. 얼굴 검출
        try:
            image, image_rgb, detections = self.detect_face(image_path)
        except Exception as e:
            print(f"⚠️ 얼굴 검출 실패: {e}")
            print("💡 얼굴이 없는 이미지로 처리합니다...")
        
        # 2. 프롬프트 생성
        prompt = self.get_emotion_prompt(emotion)
        print(f"📝 생성 프롬프트: {prompt}")
        
        # 3. Img2Img 파이프라인으로 생성
        print("🎨 Img2Img로 이미지 생성 중...")
        
        # 파이프라인 로드
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # LoRA 모델 로드
        lora_path = "lora_trained_model/final"
        if os.path.exists(lora_path):
            print(f"🎭 LoRA 모델 로드: {lora_path}")
            pipe.load_lora_weights(lora_path)
        
        if self.device == "cuda":
            pipe = pipe.to(self.device)
        
        # 원본 이미지 로드 및 전처리
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        
        # 이미지 생성 (strength: 0.7 = 원본의 30% 유지, 70% 새로 생성)
        print("🖼️ 이미지 생성 중... (약 2-3분 소요)")
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.7,  # 원본 얼굴 구조를 더 잘 유지
            guidance_scale=7.5,
            num_inference_steps=20  # 더 빠르게 (기존 30에서 20으로)
        )
        
        # 저장
        result.images[0].save(output_path)
        print(f"✅ 초상화 저장 완료: {output_path}")
        
        return result.images[0], emotion


def main():
    parser = argparse.ArgumentParser(description="빠른 셀카→초상화 변환")
    parser.add_argument("--input", type=str, required=True, help="입력 셀카 이미지 경로")
    parser.add_argument("--output", type=str, default="my_portrait_fast.png", help="출력 파일명")
    parser.add_argument("--emotion", type=str, default="joy", 
                       choices=["joy", "sadness", "anger", "fear", "surprise", "neutral", "love"],
                       help="원하는 감정")
    parser.add_argument("--lora-path", type=str, default="lora_trained_model/final", 
                       help="LoRA 모델 경로")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 변환기 초기화
    converter = QuickSelfieToPortrait(lora_model_path=args.lora_path)
    
    # 초상화 생성
    image, emotion = converter.create_portrait_fast(
        args.input, 
        args.output,
        args.emotion
    )
    
    print(f"\n🎉 완료!")
    print(f"🎭 감정: {emotion}")
    print(f"💾 저장 위치: {args.output}")

if __name__ == "__main__":
    main()





