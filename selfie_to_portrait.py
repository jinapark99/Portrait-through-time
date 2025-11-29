#!/usr/bin/env python3
"""
📸 셀카 → 감정 분석 → 초상화 생성 시스템
셀카를 입력하면 얼굴의 감정을 분석하고, 그 감정과 얼굴 구조를 유지하면서
학습된 초상화 스타일로 변환합니다.
"""

import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import pipeline
import mediapipe as mp

# 간단한 감정 분석을 위한 DeepFace 대체
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace를 설치하면 더 정확한 감정 분석이 가능합니다: pip install deepface")

class SelfieToPortrait:
    def __init__(self, lora_model_path="lora_trained_model/final"):
        """셀카를 초상화로 변환하는 시스템 초기화"""
        print("🚀 셀카→초상화 변환기 초기화 중...")
        
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 사용 디바이스: {self.device}")
        
        # MediaPipe Face Detection 초기화 (얼굴 검출용)
        print("👤 얼굴 검출 모델 로드 중...")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        print("✅ 셀카→초상화 변환기 초기화 완료!")
    
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
    
    def analyze_emotion_from_face(self, image_path):
        """얼굴에서 감정 분석"""
        print("🔍 얼굴 감정 분석 중...")
        
        if DEEPFACE_AVAILABLE:
            try:
                # DeepFace로 감정 분석
                analysis = DeepFace.analyze(
                    img_path=image_path, 
                    actions=['emotion'],
                    enforce_detection=False
                )
                
                # 결과 처리
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                emotions = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']
                
                print(f"💭 감지된 감정: {dominant_emotion}")
                print(f"📊 감정 점수: {emotions}")
                
                return self._map_emotion(dominant_emotion), emotions
            
            except Exception as e:
                print(f"⚠️ DeepFace 감정 분석 실패: {e}")
                return self._simple_emotion_analysis(image_path)
        else:
            return self._simple_emotion_analysis(image_path)
    
    def _simple_emotion_analysis(self, image_path):
        """간단한 감정 분석 (대체 방법)"""
        print("💭 간단한 감정 분석 사용 중...")
        # 기본값으로 neutral 반환
        return "neutral", {"neutral": 0.8}
    
    def _map_emotion(self, deepface_emotion):
        """DeepFace 감정을 우리 시스템의 감정으로 매핑"""
        emotion_map = {
            'happy': 'joy',
            'sad': 'sadness',
            'angry': 'anger',
            'surprise': 'surprise',
            'fear': 'fear',
            'disgust': 'disgust',
            'neutral': 'neutral'
        }
        return emotion_map.get(deepface_emotion.lower(), 'neutral')
    
    def get_emotion_prompt(self, emotion):
        """감정에 따른 프롬프트 생성"""
        emotion_prompts = {
            'joy': "a joyful portrait, happy expression, warm smile, bright eyes, cheerful",
            'sadness': "a melancholic portrait, sad expression, contemplative gaze, gentle mood",
            'anger': "a powerful portrait, intense expression, strong features, dramatic",
            'fear': "a thoughtful portrait, cautious expression, mysterious atmosphere",
            'surprise': "an expressive portrait, surprised expression, wide eyes, dynamic",
            'disgust': "a dignified portrait, composed expression, refined features",
            'neutral': "a serene portrait, calm expression, peaceful atmosphere, balanced",
            'love': "a tender portrait, gentle expression, warm atmosphere, caring eyes"
        }
        return emotion_prompts.get(emotion, emotion_prompts['neutral'])
    
    def create_portrait_simple(self, image_path, output_path="selfie_portrait.png"):
        """
        간단한 방법: 셀카 + 감정 → 초상화 생성
        (ControlNet 없이, 텍스트 프롬프트만 사용)
        """
        print(f"\n🎨 초상화 생성 시작!")
        print(f"📸 입력 이미지: {image_path}")
        
        # 1. 얼굴 검출
        image, image_rgb, detections = self.detect_face(image_path)
        
        # 2. 감정 분석
        emotion, emotion_scores = self.analyze_emotion_from_face(image_path)
        print(f"🎭 분석된 감정: {emotion}")
        
        # 3. 프롬프트 생성
        prompt = self.get_emotion_prompt(emotion)
        print(f"📝 생성 프롬프트: {prompt}")
        
        # 4. Stable Diffusion으로 이미지 생성
        print("🎨 이미지 생성 중... (LoRA 모델 사용)")
        print("💡 팁: ControlNet을 사용하면 얼굴 구조를 더 잘 유지할 수 있습니다!")
        
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        # 파이프라인 로드
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        
        # LoRA 모델 로드
        lora_path = "lora_trained_model/final"
        if os.path.exists(lora_path):
            print(f"🎭 LoRA 모델 로드: {lora_path}")
            pipe.load_lora_weights(lora_path)
        
        if self.device == "cuda":
            pipe = pipe.to(self.device)
        
        # 이미지 생성
        result = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        # 저장
        result.images[0].save(output_path)
        print(f"✅ 초상화 저장 완료: {output_path}")
        
        return result.images[0], emotion, emotion_scores
    
    def create_portrait_with_img2img(self, image_path, output_path="selfie_portrait.png"):
        """
        Img2Img 방법: 원본 이미지를 참조하여 초상화 생성
        얼굴 구조를 더 잘 유지합니다!
        """
        print(f"\n🎨 Img2Img 초상화 생성 시작!")
        print(f"📸 입력 이미지: {image_path}")
        
        # 1. 얼굴 검출
        image, image_rgb, detections = self.detect_face(image_path)
        
        # 2. 감정 분석
        emotion, emotion_scores = self.analyze_emotion_from_face(image_path)
        print(f"🎭 분석된 감정: {emotion}")
        
        # 3. 프롬프트 생성
        prompt = self.get_emotion_prompt(emotion)
        print(f"📝 생성 프롬프트: {prompt}")
        
        # 4. Img2Img 파이프라인으로 생성
        print("🎨 Img2Img로 이미지 생성 중...")
        
        from diffusers import StableDiffusionImg2ImgPipeline
        
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
        
        # 이미지 생성 (strength: 0.75 = 원본의 25% 유지, 75% 새로 생성)
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,  # 0.0 = 원본 그대로, 1.0 = 완전히 새로 생성
            guidance_scale=7.5,
            num_inference_steps=30
        )
        
        # 저장
        result.images[0].save(output_path)
        print(f"✅ 초상화 저장 완료: {output_path}")
        
        return result.images[0], emotion, emotion_scores


def main():
    parser = argparse.ArgumentParser(description="셀카를 초상화로 변환")
    parser.add_argument("--input", type=str, required=True, help="입력 셀카 이미지 경로")
    parser.add_argument("--output", type=str, default="selfie_portrait.png", help="출력 파일명")
    parser.add_argument("--method", type=str, default="img2img", 
                       choices=["simple", "img2img"],
                       help="생성 방법 (simple: 텍스트만, img2img: 원본 참조)")
    parser.add_argument("--lora-path", type=str, default="lora_trained_model/final", 
                       help="LoRA 모델 경로")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 변환기 초기화
    converter = SelfieToPortrait(lora_model_path=args.lora_path)
    
    # 초상화 생성
    if args.method == "img2img":
        image, emotion, scores = converter.create_portrait_with_img2img(
            args.input, 
            args.output
        )
    else:
        image, emotion, scores = converter.create_portrait_simple(
            args.input, 
            args.output
        )
    
    print(f"\n🎉 완료!")
    print(f"🎭 감정: {emotion}")
    print(f"📊 감정 점수: {scores}")
    print(f"💾 저장 위치: {args.output}")

if __name__ == "__main__":
    main()






