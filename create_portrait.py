#!/usr/bin/env python3
"""
🎨 간단한 셀카 → 초상화 변환기
사진을 넣으면 초상화를 생성합니다!
"""

import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import mediapipe as mp
import os

class SelfieToPortrait:
    def __init__(self):
        print("🎨 셀카 → 초상화 변환기 초기화 중...")
        
        # M1 GPU 설정 (MPS 지원)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"📱 사용 디바이스: {self.device}")
        
        # MediaPipe 얼굴 감지 초기화
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Stable Diffusion 파이프라인 로드
        print("🎨 Stable Diffusion 파이프라인 로드 중... (처음 실행 시 시간이 걸릴 수 있습니다)")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device in ["mps", "cuda"] else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)
        
        # LoRA 로드 (있는 경우)
        lora_path = "lora_trained_model/final"
        if os.path.exists(lora_path):
            print("🎭 LoRA 모델 로드 중...")
            try:
                self.pipe.load_lora_weights(lora_path)
                print("✅ LoRA 로드 성공!")
            except Exception as e:
                print(f"⚠️ LoRA 로드 실패 (계속 진행): {e}")
        else:
            print("💡 LoRA 모델을 찾을 수 없습니다. 기본 모델로 진행합니다.")
        
        print("✅ 초기화 완료!\n")
    
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
            
            # 얼굴 영역 크롭 (약간의 여유 공간 추가)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(w - x, width + padding * 2)
            height = min(h - y, height + padding * 2)
            
            face_crop = image.crop((x, y, x + width, y + height))
            
            print(f"✅ 얼굴 감지 성공: {width}x{height}")
            return face_crop, True
        else:
            print("⚠️ 얼굴을 찾을 수 없습니다. 전체 이미지를 사용합니다.")
            return image, False
    
    def analyze_emotion_simple(self, face_image):
        """간단한 감정 분석"""
        # 얼굴 랜드마크 감지
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
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
            
            # 간단한 감정 판단
            if mouth_curve < -0.005:  # 입꼬리가 위로 올라감
                emotion = "joyful"
                print("😊 감정 분석: 기쁨 (웃는 표정)")
            else:
                emotion = "neutral"
                print("😐 감정 분석: 중립")
        else:
            # 랜드마크 감지 실패 시 이미지 밝기로 감정 추정
            brightness = np.mean(img_array)
            
            if brightness > 120:  # 밝은 이미지
                emotion = "joyful"
                print("😊 감정 분석: 기쁨 (밝은 이미지로 추정)")
            else:
                emotion = "neutral"
                print("😐 감정 분석: 중립 (밝기 기반)")
        
        return emotion
    
    def create_portrait(self, selfie_path, output_path=None, emotion_override=None):
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
        
        # 감정 분석 (오버라이드가 없으면)
        if emotion_override:
            emotion = emotion_override
            print(f"🎭 사용자 지정 감정: {emotion}")
        else:
            emotion = self.analyze_emotion_simple(input_image)
        
        # 프롬프트 생성
        prompt = f"portrait painting, {emotion} expression, classical art style, high quality, detailed"
        print(f"📝 사용 프롬프트: {prompt}")
        
        # 출력 경로 설정
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(selfie_path))[0]
            output_path = f"portrait_{base_name}_{emotion}.png"
        
        # 이미지 생성
        print("🎨 이미지 생성 중... (약 1-3분 소요, GPU에 따라 다를 수 있습니다)")
        try:
            generator = torch.Generator(device=self.device)
            result = self.pipe(
                prompt=prompt,
                image=input_image,
                strength=0.7,  # 원본 이미지와의 유사도
                guidance_scale=7.5,  # 프롬프트 준수도
                num_inference_steps=20,
                generator=generator
            )
            
            # 결과 저장
            result.images[0].save(output_path)
            print(f"✅ 초상화 생성 완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 이미지 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="🎨 셀카를 초상화로 변환합니다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python create_portrait.py --input my_photo.jpg
  python create_portrait.py --input my_photo.jpg --output my_portrait.png
  python create_portrait.py --input my_photo.jpg --emotion joyful
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="입력 셀카 이미지 경로"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="출력 파일 경로 (기본값: portrait_[입력파일명]_[감정].png)"
    )
    parser.add_argument(
        "--emotion", "-e",
        type=str,
        default=None,
        choices=["joyful", "neutral", "sad", "angry", "surprised", "contemplative"],
        help="원하는 감정 (지정하지 않으면 자동 분석)"
    )
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    print("=" * 60)
    print("🎨 초상화 생성기 시작!")
    print("=" * 60)
    
    # 변환기 초기화
    converter = SelfieToPortrait()
    
    # 포트레이트 생성
    result_path = converter.create_portrait(
        args.input,
        args.output,
        args.emotion
    )
    
    if result_path:
        print("\n" + "=" * 60)
        print(f"🎉 완료! 결과 이미지: {result_path}")
        print("=" * 60)
    else:
        print("\n❌ 변환 실패")

if __name__ == "__main__":
    main()


