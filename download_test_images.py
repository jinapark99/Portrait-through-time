#!/usr/bin/env python3
"""
🌐 웹에서 감정별 이미지 다운로드 및 테스트
명확한 웃음과 슬픔 표정의 이미지들을 다운로드해서 테스트합니다.
"""

import requests
from PIL import Image
import io
import os

def download_test_images():
    """웹에서 감정별 테스트 이미지 다운로드"""
    print("🌐 웹에서 감정별 테스트 이미지 다운로드 중...")
    
    # 테스트용 이미지 URL들 (무료 이미지)
    test_images = {
        "happy_face": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&h=512&fit=crop&crop=face",
        "sad_face": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512&h=512&fit=crop&crop=face",
        "surprised_face": "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=512&h=512&fit=crop&crop=face"
    }
    
    downloaded_files = {}
    
    for emotion, url in test_images.items():
        try:
            print(f"📥 다운로드 중: {emotion}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # 이미지 로드 및 저장
            image = Image.open(io.BytesIO(response.content))
            filename = f"test_{emotion}.jpg"
            image.save(filename)
            
            downloaded_files[emotion] = filename
            print(f"✅ 다운로드 완료: {filename} ({image.size})")
            
        except Exception as e:
            print(f"❌ 다운로드 실패 ({emotion}): {e}")
    
    return downloaded_files

def test_emotion_detection():
    """다운로드한 이미지들로 감정 감지 테스트"""
    print("\n🎨 감정 감지 테스트 시작...")
    
    # 다운로드한 이미지들 확인
    test_files = {
        "happy": "test_happy_face.jpg",
        "sad": "test_sad_face.jpg", 
        "surprised": "test_surprised_face.jpg"
    }
    
    for emotion, filename in test_files.items():
        if os.path.exists(filename):
            print(f"\n🔍 테스트 중: {filename}")
            
            # 이미지 로드
            try:
                image = Image.open(filename)
                print(f"   이미지 크기: {image.size}")
                
                # 간단한 이미지 분석
                img_array = np.array(image)
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
                
                print(f"   밝기: {brightness:.2f}")
                print(f"   대비: {contrast:.2f}")
                
                # 예상 감정과 비교
                print(f"   예상 감정: {emotion}")
                
            except Exception as e:
                print(f"   ❌ 이미지 로드 실패: {e}")
        else:
            print(f"❌ 파일 없음: {filename}")

if __name__ == "__main__":
    import numpy as np
    
    # 이미지 다운로드
    downloaded = download_test_images()
    
    # 감정 감지 테스트
    test_emotion_detection()
    
    print(f"\n🎉 다운로드 완료! {len(downloaded)}개 파일")
    for emotion, filename in downloaded.items():
        print(f"   {emotion}: {filename}")





