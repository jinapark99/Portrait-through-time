#!/usr/bin/env python3
"""
🎭 텍스트 감정 분석 + LoRA 초상화 생성 시스템
텍스트를 입력하면 감정을 분석하고 해당 감정에 맞는 초상화를 생성합니다.
"""

import torch
import argparse
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline
import re

class EmotionPortraitGenerator:
    def __init__(self, lora_model_path="./final"):
        """감정 초상화 생성기 초기화"""
        print("🚀 감정 초상화 생성기 초기화 중...")
        
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 사용 디바이스: {self.device}")
        
        # 감정 분석 모델 로드 (더 간단한 모델 사용)
        print("🧠 감정 분석 모델 로드 중...")
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"⚠️ 감정 분석 모델 로드 실패: {e}")
            print("🔄 간단한 키워드 기반 감정 분석으로 대체합니다.")
            self.emotion_classifier = None
        
        # Stable Diffusion 파이프라인 로드
        print("🎨 Stable Diffusion 모델 로드 중...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 스케줄러 설정
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # LoRA 모델 로드
        if os.path.exists(lora_model_path):
            print(f"🎭 LoRA 모델 로드 중: {lora_model_path}")
            self.pipe.load_lora_weights(lora_model_path)
            print("✅ LoRA 모델 로드 완료!")
        else:
            print(f"⚠️ LoRA 모델을 찾을 수 없습니다: {lora_model_path}")
        
        # GPU로 이동
        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)
        
        print("✅ 감정 초상화 생성기 초기화 완료!")
    
    def analyze_emotion(self, text):
        """텍스트에서 감정 분석"""
        print(f"🔍 텍스트 감정 분석 중: '{text[:50]}...'")
        
        if self.emotion_classifier is not None:
            try:
                # 감정 분석 실행
                result = self.emotion_classifier(text)
                
                # 가장 높은 점수의 감정 추출
                primary_emotion = result[0]['label']
                confidence = result[0]['score']
                
                print(f"💭 감정 분석 결과: {primary_emotion} (신뢰도: {confidence:.2f})")
                
                return primary_emotion, confidence
            except Exception as e:
                print(f"⚠️ 감정 분석 오류: {e}")
                return self._keyword_based_emotion(text)
        else:
            return self._keyword_based_emotion(text)
    
    def _keyword_based_emotion(self, text):
        """키워드 기반 감정 분석 (대체 방법)"""
        text_lower = text.lower()
        
        # 감정 키워드 매핑
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'amazing', 'wonderful', 'great', 'fantastic', 'delighted', 'cheerful'],
            'love': ['love', 'adore', 'cherish', 'affection', 'romantic', 'caring', 'tender'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'annoyed', 'irritated', 'frustrated'],
            'fear': ['afraid', 'scared', 'fear', 'worried', 'anxious', 'nervous', 'terrified'],
            'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'unhappy', 'down', 'blue'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'wow', 'incredible'],
            'disgust': ['disgusted', 'revolted', 'sick', 'gross', 'nasty', 'repulsive']
        }
        
        # 감정 점수 계산
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            # 가장 높은 점수의 감정 선택
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(emotion_scores[primary_emotion] / len(text.split()), 1.0)
        else:
            primary_emotion = 'neutral'
            confidence = 0.5
        
        print(f"💭 키워드 기반 감정 분석 결과: {primary_emotion} (신뢰도: {confidence:.2f})")
        
        return primary_emotion, confidence
    
    def get_emotion_style(self, emotion):
        """감정에 따른 초상화 스타일 매핑"""
        emotion_styles = {
            'joy': "bright, cheerful, radiant smile, warm lighting, vibrant colors, optimistic expression",
            'love': "gentle, warm, tender expression, soft lighting, romantic atmosphere, caring eyes",
            'anger': "intense, powerful, dramatic lighting, strong facial features, bold expression, dynamic pose",
            'fear': "mysterious, shadowy, contemplative, subtle lighting, cautious expression, introspective",
            'sadness': "melancholic, thoughtful, gentle expression, soft lighting, contemplative mood, tender",
            'surprise': "animated, expressive, bright eyes, dynamic pose, energetic, lively expression",
            'disgust': "serious, composed, dignified expression, clean lighting, refined features, elegant",
            'neutral': "calm, peaceful, balanced expression, natural lighting, composed, serene"
        }
        
        return emotion_styles.get(emotion, emotion_styles['neutral'])
    
    def generate_portrait(self, text, output_path="generated_portrait.png", num_images=1):
        """텍스트에서 감정을 분석하고 초상화 생성"""
        print(f"\n🎭 초상화 생성 시작!")
        print(f"📝 입력 텍스트: '{text}'")
        
        # 1. 감정 분석
        emotion, confidence = self.analyze_emotion(text)
        
        # 2. 감정에 따른 스타일 결정
        style = self.get_emotion_style(emotion)
        
        # 3. 프롬프트 생성 (최소화)
        full_prompt = f"a portrait, {emotion}"
        
        print(f"🎨 생성 프롬프트: '{full_prompt}'")
        
        # 4. 이미지 생성
        print("🖼️ 이미지 생성 중...")
        images = self.pipe(
            full_prompt,
            num_inference_steps=20,
            num_images_per_prompt=num_images,
            guidance_scale=7.5
        ).images
        
        # 5. 이미지 저장
        if num_images == 1:
            images[0].save(output_path)
            print(f"✅ 이미지 저장 완료: {output_path}")
        else:
            for i, image in enumerate(images):
                filename = f"generated_portrait_{i+1}.png"
                image.save(filename)
                print(f"✅ 이미지 저장 완료: {filename}")
        
        return images, emotion, confidence
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n🎭 감정 초상화 생성기 대화형 모드")
        print("💡 텍스트를 입력하면 감정을 분석하고 초상화를 생성합니다.")
        print("🚪 종료하려면 'quit' 또는 'exit'를 입력하세요.")
        
        while True:
            try:
                text = input("\n📝 텍스트를 입력하세요: ").strip()
                
                if text.lower() in ['quit', 'exit', '종료']:
                    print("👋 감정 초상화 생성기를 종료합니다.")
                    break
                
                if not text:
                    print("⚠️ 텍스트를 입력해주세요.")
                    continue
                
                # 이미지 생성
                images, emotion, confidence = self.generate_portrait(text)
                
                print(f"\n🎉 완료! 감정: {emotion} (신뢰도: {confidence:.2f})")
                
            except KeyboardInterrupt:
                print("\n👋 감정 초상화 생성기를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description="텍스트 감정 분석 + LoRA 초상화 생성")
    parser.add_argument("--text", type=str, help="분석할 텍스트")
    parser.add_argument("--output", type=str, default="generated_portrait.png", help="출력 파일명")
    parser.add_argument("--lora-path", type=str, default="./final", help="LoRA 모델 경로")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")
    parser.add_argument("--num-images", type=int, default=1, help="생성할 이미지 개수")
    
    args = parser.parse_args()
    
    # 생성기 초기화
    generator = EmotionPortraitGenerator(lora_model_path=args.lora_path)
    
    if args.interactive:
        # 대화형 모드
        generator.interactive_mode()
    elif args.text:
        # 단일 텍스트 처리
        images, emotion, confidence = generator.generate_portrait(
            args.text, 
            args.output, 
            args.num_images
        )
        print(f"\n🎉 완료! 감정: {emotion} (신뢰도: {confidence:.2f})")
    else:
        print("❌ 텍스트를 입력하거나 --interactive 옵션을 사용하세요.")
        print("💡 예시: python emotion_portrait_generator.py --text 'I am so happy today!'")

if __name__ == "__main__":
    main()
