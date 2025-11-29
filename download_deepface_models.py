#!/usr/bin/env python3
"""
🔧 DeepFace 모델 미리 다운로드 스크립트
DeepFace가 사용하는 모든 모델을 미리 다운로드해서 나중에 빠르게 사용할 수 있게 합니다.
"""

import os
import time
from deepface import DeepFace

def download_deepface_models():
    """DeepFace 모델들을 미리 다운로드"""
    print("🚀 DeepFace 모델 다운로드 시작...")
    print("=" * 50)
    
    # 테스트용 이미지 생성 (1x1 픽셀)
    import numpy as np
    from PIL import Image
    
    # 더미 이미지 생성
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    Image.fromarray(dummy_img).save("dummy_test.jpg")
    
    print("📸 테스트용 더미 이미지 생성 완료")
    
    # 각 모델별로 다운로드
    models = [
        "VGG-Face",
        "Facenet", 
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib"
    ]
    
    backends = [
        "opencv",
        "retinaface", 
        "mtcnn",
        "ssd",
        "dlib"
    ]
    
    print(f"\n🧠 감정 분석 모델 다운로드 중...")
    try:
        # 감정 분석 모델 다운로드
        result = DeepFace.analyze(
            img_path="dummy_test.jpg",
            actions=['emotion'],
            enforce_detection=False,
            detector_backend="opencv"
        )
        print("✅ 감정 분석 모델 다운로드 완료!")
        print(f"📊 감정 분석 결과: {result}")
    except Exception as e:
        print(f"⚠️ 감정 분석 모델 다운로드 실패: {e}")
    
    print(f"\n👤 얼굴 인식 모델 다운로드 중...")
    for model in models[:3]:  # 처음 3개만 테스트
        try:
            print(f"  📥 {model} 모델 다운로드 중...")
            DeepFace.represent(
                img_path="dummy_test.jpg",
                model_name=model,
                enforce_detection=False
            )
            print(f"  ✅ {model} 모델 다운로드 완료!")
        except Exception as e:
            print(f"  ⚠️ {model} 모델 다운로드 실패: {e}")
    
    print(f"\n🔍 얼굴 검출 모델 다운로드 중...")
    for backend in backends[:2]:  # 처음 2개만 테스트
        try:
            print(f"  📥 {backend} 백엔드 다운로드 중...")
            DeepFace.extract_faces(
                img_path="dummy_test.jpg",
                detector_backend=backend,
                enforce_detection=False
            )
            print(f"  ✅ {backend} 백엔드 다운로드 완료!")
        except Exception as e:
            print(f"  ⚠️ {backend} 백엔드 다운로드 실패: {e}")
    
    # 더미 파일 삭제
    if os.path.exists("dummy_test.jpg"):
        os.remove("dummy_test.jpg")
        print("\n🗑️ 테스트 파일 정리 완료")
    
    print("\n🎉 DeepFace 모델 다운로드 완료!")
    print("이제 실제 셀카 분석이 훨씬 빨라질 것입니다!")

if __name__ == "__main__":
    download_deepface_models()





