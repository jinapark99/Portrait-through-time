#!/usr/bin/env python3
"""
🚀 순수 PyTorch 셀카→초상화 변환기
TensorFlow 없이 PyTorch만 사용하는 깔끔한 버전입니다.
"""

import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import mediapipe as mp

class PurePyTorchSelfieToPortrait:
    def __init__(self, lora_model_path="lora_trained_model/final"):
        """순수 PyTorch 셀카→초상화 변환기 초기화"""
        print("🚀 순수 PyTorch 셀카→초상화 변환기 초기화 중...")
        
        # M1/M2 GPU 사용 설정
        if torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
            print("🍎 Apple Silicon GPU (MPS) 사용!")
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
        
        print("✅ 순수 PyTorch 변환기 초기화 완료!")
    
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
    
    def analyze_emotion_simple(self, image_path):
        """간단한 감정 분석 (색상 기반)"""
        print("🔍 간단한 감정 분석 중...")
        
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 얼굴 검출
            results = self.face_detection.process(image_rgb)
            
            if results.detections:
                # 첫 번째 얼굴 영역 분석
                face = results.detections[0]
                bbox = face.location_data.relative_bounding_box
                
                h, w = image_rgb.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # 얼굴 영역 추출
                face_crop = image_rgb[y:y+height, x:x+width]
                
                # 색상 분석
                avg_color = np.mean(face_crop, axis=(0, 1))
                brightness = np.mean(avg_color)
                
                # 간단한 감정 추정
                if brightness > 160:
                    emotion = "joy"
                elif brightness < 120:
                    emotion = "sadness"
                else:
                    emotion = "neutral"
                    
                print(f"💭 분석된 감정: {emotion} (밝기: {brightness:.1f})")
            else:
                emotion = "joy"  # 기본값
                print("💭 얼굴을 찾을 수 없어 기본값 'joy' 사용")
            
            return emotion, {emotion: 0.8}
            
        except Exception as e:
            print(f"⚠️ 감정 분석 실패: {e}")
            print("💡 기본값 'joy' 사용")
            return "joy", {"joy": 0.8}
    
    def get_emotion_prompt(self, emotion):
        """감정에 따른 프롬프트 생성"""
        emotion_prompts = {
            'joy': "a joyful portrait, happy expression, warm smile, bright eyes, cheerful, beautiful lighting, vibrant colors, masterpiece, high quality",
            'sadness': "a melancholic portrait, sad expression, contemplative gaze, gentle mood, soft lighting, muted colors, artistic, high quality",
            'anger': "a powerful portrait, intense expression, strong features, dramatic lighting, bold colors, dynamic, high quality",
            'fear': "a thoughtful portrait, cautious expression, mysterious atmosphere, subtle lighting, cool tones, introspective, high quality",
            'surprise': "an expressive portrait, surprised expression, wide eyes, dynamic pose, bright lighting, energetic, high quality",
            'disgust': "a dignified portrait, composed expression, refined features, elegant lighting, sophisticated, classical, high quality",
            'neutral': "a serene portrait, calm expression, peaceful atmosphere, balanced lighting, natural colors, tranquil, high quality",
            'love': "a tender portrait, gentle expression, warm atmosphere, caring eyes, romantic lighting, soft colors, intimate, high quality"
        }
        return emotion_prompts.get(emotion, emotion_prompts['joy'])
    
    def create_portrait_pure(self, image_path, output_path="my_portrait_pure.png"):
        """
        순수 PyTorch 초상화 생성
        """
        print(f"\n🎨 순수 PyTorch 초상화 생성 시작!")
        print(f"📸 입력 이미지: {image_path}")
        
        # 1. 감정 분석
        emotion, emotion_scores = self.analyze_emotion_simple(image_path)
        
        # 2. 프롬프트 생성
        prompt = self.get_emotion_prompt(emotion)
        print(f"📝 생성 프롬프트: {prompt}")
        
        # 3. PyTorch 파이프라인 로드
        print("🎨 PyTorch 파이프라인 로드 중...")
        
        # 파이프라인 로드
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
        
        # 디바이스로 이동
        pipe = pipe.to(self.device)
        print(f"🚀 {self.device.upper()}로 이동 완료!")
        
        # 원본 이미지 로드 및 전처리
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        
        # 이미지 생성
        print("🖼️ 이미지 생성 중... (진행률 표시)")
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.7,  # 원본 얼굴 구조 유지
            guidance_scale=7.5,
            num_inference_steps=20,  # 빠른 생성
            generator=torch.Generator(device=self.device)
        )
        
        # 저장
        result.images[0].save(output_path)
        print(f"✅ 초상화 저장 완료: {output_path}")
        
        return result.images[0], emotion, emotion_scores


def main():
    parser = argparse.ArgumentParser(description="순수 PyTorch 셀카→초상화 변환")
    parser.add_argument("--input", type=str, required=True, help="입력 셀카 이미지 경로")
    parser.add_argument("--output", type=str, default="my_portrait_pure.png", help="출력 파일명")
    parser.add_argument("--lora-path", type=str, default="lora_trained_model/final", 
                       help="LoRA 모델 경로")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 변환기 초기화
    converter = PurePyTorchSelfieToPortrait(lora_model_path=args.lora_path)
    
    # 초상화 생성
    image, emotion, scores = converter.create_portrait_pure(
        args.input, 
        args.output
    )
    
    print(f"\n🎉 완료!")
    print(f"🎭 감정: {emotion}")
    print(f"📊 감정 점수: {scores}")
    print(f"💾 저장 위치: {args.output}")

if __name__ == "__main__":
    main()





