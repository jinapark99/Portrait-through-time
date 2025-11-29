#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메트로폴리탄 미술관 초상화 추가 수집
기존 253개에서 더 많은 초상화 수집
"""

import os
import requests
import pandas as pd
import time
import json
import random
import re
from datetime import datetime

# 설정
SAVE_DIR = "data/smart_portraits"
METADATA_DIR = "data/smart_metadata"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# 메트로폴리탄 미술관 API 설정
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

# User-Agent 리스트
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def human_delay():
    """인간적인 지연"""
    delay = random.uniform(1.0, 2.0)
    time.sleep(delay)

def search_portraits_extended(limit=5000):
    """확장된 키워드로 초상화 검색"""
    
    print(f"🔍 확장된 키워드로 초상화 검색 중... (목표: {limit}개)")
    
    # 더 많은 키워드들
    keywords = [
        # 기본 초상화 키워드
        "portrait", "self-portrait", "portrait of a man", "portrait of a woman",
        "portrait of a lady", "portrait of a gentleman", "bust", "head",
        
        # 미술가별 키워드
        "rembrandt", "van gogh", "picasso", "monet", "renoir", "degas",
        "titian", "raphael", "michelangelo", "leonardo", "caravaggio",
        "velazquez", "rubens", "van dyck", "hals", "vermeer",
        
        # 시대별 키워드
        "renaissance portrait", "baroque portrait", "classical portrait",
        "romantic portrait", "impressionist portrait", "modern portrait",
        
        # 스타일별 키워드
        "formal portrait", "casual portrait", "official portrait",
        "royal portrait", "noble portrait", "bourgeois portrait",
        
        # 다국어 키워드
        "retrato", "porträt", "portrait français", "ritratto",
        
        # 구체적인 초상화 유형
        "portrait of a child", "portrait of a family", "portrait of a couple",
        "portrait of a king", "portrait of a queen", "portrait of a merchant",
        "portrait of a scholar", "portrait of a priest", "portrait of a soldier",
        
        # 미술사적 용어
        "portrait painting", "portrait art", "portrait work", "portrait study",
        "portrait sketch", "portrait drawing", "portrait miniature",
        
        # 추가 유명 미술가들
        "holbein", "durer", "botticelli", "memling", "christus",
        "el greco", "goya", "delacroix", "ingres", "courbet",
        "manet", "cezanne", "gauguin", "toulouse-lautrec", "klimt",
        
        # 현대 미술가들
        "warhol", "bacon", "freud", "hockney", "koons",
        
        # 특수 초상화 유형
        "equestrian portrait", "coronation portrait", "wedding portrait",
        "funeral portrait", "memorial portrait", "commemorative portrait"
    ]
    
    all_object_ids = set()
    
    for keyword in keywords:
        try:
            print(f"  🔎 '{keyword}' 검색 중...")
            
            search_url = f"{MET_API_BASE}/search"
            params = {
                'q': keyword,
                'hasImages': 'true'
            }
            
            headers = {'User-Agent': get_random_user_agent()}
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                object_ids = data.get('objectIDs', [])
                all_object_ids.update(object_ids)
                print(f"    ✅ {len(object_ids)} 개 발견 (총 {len(all_object_ids)} 개 수집)")
            else:
                print(f"    ❌ 검색 실패: {response.status_code}")
            
            # 키워드 간 지연
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception as e:
            print(f"    ❌ 검색 오류: {e}")
            continue
    
    print(f"🎉 총 {len(all_object_ids)} 개의 고유한 작품 발견!")
    return list(all_object_ids)[:limit]

def is_portrait_advanced(obj_data):
    """고급 초상화 판단 로직"""
    
    # 모든 텍스트 필드 수집
    text_fields = []
    
    # 제목
    title = obj_data.get('title', '')
    if title:
        text_fields.append(title.lower())
    
    # 분류
    classification = obj_data.get('classification', '')
    if classification:
        text_fields.append(classification.lower())
    
    # 부서
    department = obj_data.get('department', '')
    if department:
        text_fields.append(department.lower())
    
    # 문화/시대
    culture = obj_data.get('culture', '')
    if culture:
        text_fields.append(culture.lower())
    
    # 기간
    period = obj_data.get('period', '')
    if period:
        text_fields.append(period.lower())
    
    # 매체
    medium = obj_data.get('medium', '')
    if medium:
        text_fields.append(medium.lower())
    
    # 태그 처리
    tags = obj_data.get('tags', [])
    if tags and isinstance(tags, list):
        tag_texts = [t.get('term', '') if isinstance(t, dict) else str(t) for t in tags]
        text_fields.append(' '.join(tag_texts).lower())
    
    # 모든 텍스트 합치기
    all_text = ' '.join(text_fields)
    
    # 초상화 키워드 (포함되어야 함)
    portrait_keywords = [
        'portrait', 'self-portrait', 'bust', 'head', 'face',
        'portrait of', 'portrait painting', 'portrait art',
        'retrato', 'porträt', 'ritratto', 'portrait français'
    ]
    
    # 제외 키워드 (포함되면 안됨)
    exclude_keywords = [
        'landscape', 'still life', 'nature', 'animal', 'bird', 'dog', 'cat',
        'building', 'architecture', 'interior', 'exterior', 'scene',
        'mythology', 'religious', 'biblical', 'allegory', 'symbol',
        'decorative', 'ornament', 'furniture', 'ceramic', 'textile',
        'weapon', 'armor', 'coin', 'medal', 'jewelry', 'vase',
        'sculpture', 'statue', 'relief', 'carving', 'engraving',
        'print', 'drawing', 'sketch', 'study', 'preparatory'
    ]
    
    # 초상화 키워드 확인
    has_portrait_keyword = any(keyword in all_text for keyword in portrait_keywords)
    
    # 제외 키워드 확인
    has_exclude_keyword = any(keyword in all_text for keyword in exclude_keywords)
    
    # 초상화 판단
    is_portrait = has_portrait_keyword and not has_exclude_keyword
    
    return is_portrait

def download_portrait_image(obj_data, session):
    """초상화 이미지 다운로드"""
    
    object_id = obj_data.get('objectID', '')
    title = obj_data.get('title', 'Unknown')
    artist = obj_data.get('artistDisplayName', 'Unknown')
    
    # 이미지 URL 찾기
    primary_image = obj_data.get('primaryImage', '')
    if not primary_image:
        return False, 0, "이미지 URL 없음"
    
    try:
        # 파일명 생성
        safe_title = re.sub(r"[^\w\-\.]+", "_", title)[:50]
        safe_artist = re.sub(r"[^\w\-\.]+", "_", artist)[:30]
        filename = f"{object_id}_{safe_artist}_{safe_title}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # 이미지 다운로드
        headers = {'User-Agent': get_random_user_agent()}
        response = session.get(primary_image, headers=headers, timeout=30)
        
        if response.status_code == 200 and response.content:
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            return True, file_size_mb, filename
        else:
            return False, 0, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, 0, str(e)

def main():
    print("🏛️ === 메트로폴리탄 미술관 초상화 추가 수집 ===")
    print("📋 기존 253개에서 더 많은 초상화 수집")
    print("=" * 60)
    
    # 1단계: 확장된 키워드로 검색
    all_object_ids = search_portraits_extended(limit=5000)
    
    if not all_object_ids:
        print("❌ 검색 결과가 없습니다.")
        return
    
    # 2단계: 다운로드
    print(f"\n📥 {len(all_object_ids)}개 작품 분석 및 다운로드 시작...")
    
    session = requests.Session()
    results = []
    downloaded_count = 0
    failed_count = 0
    not_portrait_count = 0
    api_error_count = 0
    
    for i, object_id in enumerate(all_object_ids):
        print(f"[{i+1}/{len(all_object_ids)}] 분석 중: ID {object_id}")
        
        try:
            # 작품 상세 정보 가져오기
            object_url = f"{MET_API_BASE}/objects/{object_id}"
            headers = {'User-Agent': get_random_user_agent()}
            response = session.get(object_url, headers=headers, timeout=20)
            
            if response.status_code == 200:
                obj_data = response.json()
                
                # 초상화 판단
                if is_portrait_advanced(obj_data):
                    title = obj_data.get('title', 'Unknown')
                    artist = obj_data.get('artistDisplayName', 'Unknown')
                    print(f"  🎨 {artist} - {title}")
                    
                    # 이미지 다운로드
                    success, file_size, result_msg = download_portrait_image(obj_data, session)
                    
                    if success:
                        print(f"  ✅ 다운로드 완료: {result_msg} ({file_size:.1f}MB)")
                        downloaded_count += 1
                        results.append({
                            "object_id": object_id,
                            "title": title,
                            "artist": artist,
                            "filename": result_msg,
                            "file_size_mb": file_size,
                            "status": "success"
                        })
                    else:
                        print(f"  ❌ 다운로드 실패: {result_msg}")
                        failed_count += 1
                        results.append({
                            "object_id": object_id,
                            "title": title,
                            "artist": artist,
                            "status": "failed",
                            "reason": result_msg
                        })
                else:
                    print(f"  ⏭️ 초상화가 아님 - 건너뜀")
                    not_portrait_count += 1
                    results.append({
                        "object_id": object_id,
                        "title": obj_data.get('title', 'Unknown'),
                        "artist": obj_data.get('artistDisplayName', 'Unknown'),
                        "status": "not_portrait"
                    })
            else:
                print(f"  ❌ 상세 정보를 가져올 수 없음")
                api_error_count += 1
                results.append({
                    "object_id": object_id,
                    "status": "api_error",
                    "reason": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            api_error_count += 1
            results.append({
                "object_id": object_id,
                "status": "error",
                "reason": str(e)
            })
        
        # 진행 상황 표시 (매 50개마다)
        if (i + 1) % 50 == 0:
            print(f"\n📊 진행 상황: {i+1:,}/{len(all_object_ids):,} 완료")
            print(f"  ✅ 성공: {downloaded_count:,}개")
            print(f"  ❌ 실패: {failed_count:,}개")
            print(f"  ⏭️ 초상화 아님: {not_portrait_count:,}개")
            print(f"  🚫 API 오류: {api_error_count:,}개")
            print(f"  📈 초상화 발견율: {(downloaded_count/(i+1)*100):.1f}%")
            
            # 예상 완료 시간 계산
            if downloaded_count > 0:
                total_size_mb = sum([r.get('file_size_mb', 0) for r in results])
                avg_size_mb = total_size_mb / downloaded_count
                remaining = len(all_object_ids) - (i + 1)
                estimated_remaining_size = remaining * avg_size_mb
                print(f"  💾 현재 총 크기: {total_size_mb:.1f}MB")
                print(f"  🔮 예상 최종 크기: {total_size_mb + estimated_remaining_size:.1f}MB")
        
        # API 오류가 많으면 휴식
        if api_error_count > 10:
            print(f"  😴 API 오류가 많아 30초 휴식...")
            time.sleep(30)
            api_error_count = 0
        
        human_delay()
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(METADATA_DIR, f"metropolitan_portraits_extended_{timestamp}.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 최종 통계
    total_size = sum([r.get('file_size_mb', 0) for r in results if r.get('status') == 'success'])
    
    print(f"\n🎉 === 메트로폴리탄 초상화 추가 수집 완료 ===")
    print(f"✅ 총 다운로드: {downloaded_count}개")
    print(f"❌ 다운로드 실패: {failed_count}개")
    print(f"⏭️ 초상화 아님: {not_portrait_count}개")
    print(f"🚫 API 오류: {api_error_count}개")
    print(f"📁 이미지 저장: {SAVE_DIR}")
    print(f"📄 메타데이터: {csv_path}")
    print(f"💾 총 파일 크기: {total_size:.1f}MB")

if __name__ == "__main__":
    main()
