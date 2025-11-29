#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초상화 작품의 작가 감정 상태 분석
GPT-4를 사용하여 작가가 작품을 그릴 당시의 심리 상태와 감정을 분석
"""

import os
import pandas as pd
import json
import time
from openai import OpenAI
from datetime import datetime

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 감정 차원 정의
EMOTION_DIMENSIONS = [
    "melancholic",      # 우울한
    "joyful",           # 기쁜
    "anxious",          # 불안한
    "confident",        # 자신감 있는
    "contemplative",    # 사색적인
    "energetic",        # 활기찬
    "lonely",           # 고독한
    "peaceful",         # 평온한
    "turbulent",        # 격동적인
    "serene"            # 고요한
]

def analyze_artist_emotion(artist, title, year, culture="", period=""):
    """
    작가가 작품을 그릴 당시의 감정 상태를 분석
    """
    
    # 프롬프트 구성
    prompt = f"""
작가: {artist if artist else "Unknown"}
작품: {title}
제작 연도: {year if year else "Unknown"}
문화권: {culture if culture else "Unknown"}
시대: {period if period else "Unknown"}

이 작가가 이 작품을 그릴 당시의 심리 상태와 삶의 맥락을 분석해주세요.

**중요 규칙:**
1. 역사적으로 검증 가능한 사실만 사용하세요
2. 작가가 유명하고 충분한 역사적 기록이 있으면 구체적으로 분석
3. 작가가 무명이거나 정보가 부족하면 confidence를 "unknown"으로 표시
4. 추측이 들어가면 confidence를 낮춰주세요

다음 JSON 형식으로 정확하게 답변해주세요:
{{
  "life_context": "작가의 당시 상황 설명 (1-2문장, 정보 없으면 'No historical records available')",
  "confidence": "high/medium/low/unknown",
  "emotion_scores": {{
    "melancholic": 0.0,
    "joyful": 0.0,
    "anxious": 0.0,
    "confident": 0.0,
    "contemplative": 0.0,
    "energetic": 0.0,
    "lonely": 0.0,
    "peaceful": 0.0,
    "turbulent": 0.0,
    "serene": 0.0
  }}
}}

각 감정은 0.0~1.0 사이 값으로 평가해주세요.
정보가 전혀 없으면 confidence: "unknown"으로 하고 모든 감정을 0.0으로 설정하세요.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 미술사 전문가입니다. 역사적 사실에 기반한 정확한 분석만 제공하세요. 추측은 피하고, 확실하지 않으면 confidence를 낮추거나 unknown으로 표시하세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # 낮은 온도로 일관성 있는 답변
            response_format={"type": "json_object"}
        )
        
        # JSON 파싱
        content = response.choices[0].message.content
        result = json.loads(content)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    print("🎨 === 초상화 감정 분석 시작 ===")
    print("📋 GPT-4를 사용하여 작가의 감정 상태 분석")
    print("=" * 60)
    
    # API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
        print("\n설정 방법:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # CSV 읽기
    csv_path = "data/smart_metadata/smart_portraits_final.csv"
    print(f"\n📂 CSV 읽는 중: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ 총 {len(df)}개 작품 발견")
    
    # 샘플 100개만 처리
    sample_size = 100
    df_sample = df.head(sample_size)
    print(f"\n🎯 샘플 {sample_size}개로 테스트 진행")
    
    results = []
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    for idx, row in df_sample.iterrows():
        print(f"\n[{idx+1}/{sample_size}] 분석 중...")
        print(f"  작가: {row.get('artist', 'Unknown')}")
        print(f"  작품: {row.get('title', 'Unknown')}")
        
        # 감정 분석
        analysis = analyze_artist_emotion(
            artist=row.get('artist', ''),
            title=row.get('title', ''),
            year=row.get('object_date', ''),
            culture=row.get('culture', ''),
            period=row.get('period', '')
        )
        
        if analysis['success']:
            data = analysis['data']
            print(f"  ✅ 분석 완료 - Confidence: {data.get('confidence', 'unknown')}")
            print(f"  📝 맥락: {data.get('life_context', 'N/A')[:80]}...")
            
            # 감정 점수 출력
            emotions = data.get('emotion_scores', {})
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  😊 주요 감정: {', '.join([f'{k}={v:.2f}' for k, v in top_emotions])}")
            
            results.append({
                'object_id': row.get('object_id'),
                'artist': row.get('artist', ''),
                'title': row.get('title', ''),
                'year': row.get('object_date', ''),
                'life_context': data.get('life_context', ''),
                'confidence': data.get('confidence', 'unknown'),
                **{f'emotion_{k}': v for k, v in emotions.items()}
            })
            success_count += 1
        else:
            print(f"  ❌ 실패: {analysis.get('error', 'Unknown error')}")
            results.append({
                'object_id': row.get('object_id'),
                'artist': row.get('artist', ''),
                'title': row.get('title', ''),
                'year': row.get('object_date', ''),
                'life_context': 'Analysis failed',
                'confidence': 'error',
                **{f'emotion_{k}': 0.0 for k in EMOTION_DIMENSIONS}
            })
            fail_count += 1
        
        # 진행 상황 표시 (매 10개마다)
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (sample_size - idx - 1)
            print(f"\n📊 진행 상황:")
            print(f"  ✅ 성공: {success_count}개")
            print(f"  ❌ 실패: {fail_count}개")
            print(f"  ⏱️  소요 시간: {elapsed/60:.1f}분")
            print(f"  🔮 예상 남은 시간: {remaining/60:.1f}분")
        
        # API 제한 방지 (너무 빠르게 요청하지 않기)
        time.sleep(1)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/smart_metadata/emotions_analysis_{timestamp}.csv"
    
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_path, index=False, encoding='utf-8')
    
    # 최종 통계
    total_time = time.time() - start_time
    
    print(f"\n🎉 === 분석 완료 ===")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {fail_count}개")
    print(f"⏱️  총 소요 시간: {total_time/60:.1f}분")
    print(f"📄 결과 저장: {output_path}")
    
    # Confidence 분포 확인
    print(f"\n📊 Confidence 분포:")
    confidence_dist = df_result['confidence'].value_counts()
    for conf, count in confidence_dist.items():
        print(f"  {conf}: {count}개 ({count/len(df_result)*100:.1f}%)")
    
    # 고신뢰도 데이터만 필터링
    high_conf = df_result[df_result['confidence'] == 'high']
    print(f"\n⭐ 고신뢰도 (high) 데이터: {len(high_conf)}개")
    if len(high_conf) > 0:
        print("  샘플:")
        for _, row in high_conf.head(3).iterrows():
            print(f"  - {row['artist']}: {row['title']}")

if __name__ == "__main__":
    main()

