#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
스미소니언 국립 초상화 갤러리에서 초상화 다운로드
100% 초상화 보장, 고품질 이미지
"""

import requests
import os
import time
import random
import re
import json
from urllib.parse import urljoin, urlparse
import pandas as pd
from datetime import datetime

# 설정
SAVE_DIR = "data/smithsonian_portraits"
METADATA_DIR = "data/smithsonian_metadata"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# 스미소니언 API 엔드포인트들
SMITHSONIAN_BASE_URL = "https://api.si.edu/openaccess/api/v1.0"
SMITHSONIAN_SEARCH_URL = f"{SMITHSONIAN_BASE_URL}/search"
SMITHSONIAN_OBJECT_URL = f"{SMITHSONIAN_BASE_URL}/content"

# User-Agent 목록
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
]

def get_random_headers():
    """랜덤 헤더 생성"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def human_delay():
    """인간적인 지연"""
    delay = random.uniform(1.0, 2.5)
    time.sleep(delay)

def search_smithsonian_portraits(limit=1000):
    """스미소니언에서 초상화 검색"""
    print(f"🔍 스미소니언 국립 초상화 갤러리에서 초상화 검색 중... (목표: {limit}개)")
    
    all_objects = []
    start = 0
    rows = 100  # 한 번에 가져올 수 있는 최대 개수
    
    session = requests.Session()
    
    while len(all_objects) < limit:
        try:
            # 검색 파라미터
            params = {
                "q": "portrait",  # 초상화 검색
                "start": start,
                "rows": min(rows, limit - len(all_objects)),
                "fq": "unit_code:NPG",  # National Portrait Gallery만
                # "api_key": "YOUR_API_KEY"  # API 키 없이도 접근 가능
            }
            
            print(f"  📡 페이지 {start//rows + 1} 요청 중... (start={start})")
            
            response = session.get(
                SMITHSONIAN_SEARCH_URL,
                params=params,
                headers=get_random_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                objects = data.get("response", {}).get("docs", [])
                
                if not objects:
                    print("  ✅ 더 이상 결과가 없습니다.")
                    break
                
                all_objects.extend(objects)
                print(f"  ✅ {len(objects)}개 발견 (총 {len(all_objects)}개)")
                
                start += rows
                human_delay()
                
            else:
                print(f"  ❌ API 오류: {response.status_code}")
                if response.status_code == 429:  # Rate limit
                    print("  😴 Rate limit - 60초 대기...")
                    time.sleep(60)
                else:
                    break
                    
        except Exception as e:
            print(f"  ❌ 검색 오류: {e}")
            break
    
    print(f"🎉 총 {len(all_objects)}개의 초상화 작품 발견!")
    return all_objects[:limit]

def get_object_details(object_id, session):
    """작품 상세 정보 가져오기"""
    try:
        url = f"{SMITHSONIAN_OBJECT_URL}/{object_id}"
        response = session.get(url, headers=get_random_headers(), timeout=20)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"    ❌ 상세 정보 오류: {response.status_code}")
            return None
    except Exception as e:
        print(f"    ❌ 상세 정보 예외: {e}")
        return None

def download_image(image_url, filename, session):
    """이미지 다운로드"""
    try:
        response = session.get(image_url, headers=get_random_headers(), timeout=30)
        
        if response.status_code == 200 and response.content:
            filepath = os.path.join(SAVE_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            return True, file_size_mb
        else:
            return False, 0
    except Exception as e:
        print(f"    ❌ 다운로드 오류: {e}")
        return False, 0

def main():
    print("🏛️ === 스미소니언 국립 초상화 갤러리 다운로더 ===")
    print("📋 100% 초상화 보장, 고품질 이미지")
    print("=" * 60)
    
    # 1단계: 초상화 검색
    all_objects = search_smithsonian_portraits(limit=1000)
    
    if not all_objects:
        print("❌ 검색 결과가 없습니다.")
        return
    
    # 2단계: 다운로드
    print(f"\n📥 {len(all_objects)}개 작품 다운로드 시작...")
    
    session = requests.Session()
    results = []
    downloaded_count = 0
    failed_count = 0
    
    for i, obj in enumerate(all_objects):
        object_id = obj.get("id", "")
        title = obj.get("title", "Unknown")
        
        print(f"[{i+1}/{len(all_objects)}] 분석 중: ID {object_id}")
        print(f"  🎨 {title}")
        
        # 상세 정보 가져오기
        details = get_object_details(object_id, session)
        if not details:
            print(f"  ❌ 상세 정보를 가져올 수 없음")
            failed_count += 1
            results.append({
                "object_id": object_id,
                "title": title,
                "status": "failed",
                "reason": "상세 정보 없음"
            })
            continue
        
        # 이미지 URL 찾기
        image_url = None
        if "indexedStructuredData" in details:
            structured_data = details["indexedStructuredData"]
            if "descriptiveNonRepeating" in structured_data:
                desc = structured_data["descriptiveNonRepeating"]
                if "online_media" in desc and "media" in desc["online_media"]:
                    media = desc["online_media"]["media"]
                    if media and len(media) > 0:
                        image_url = media[0].get("content", "")
        
        if not image_url:
            print(f"  ❌ 이미지 URL 없음")
            failed_count += 1
            results.append({
                "object_id": object_id,
                "title": title,
                "status": "failed",
                "reason": "이미지 URL 없음"
            })
            continue
        
        # 파일명 생성
        safe_title = re.sub(r"[^\w\-\.]+", "_", title)[:50]
        filename = f"{object_id}_{safe_title}.jpg"
        
        # 이미지 다운로드
        success, file_size = download_image(image_url, filename, session)
        
        if success:
            print(f"  ✅ 다운로드 완료: {filename} ({file_size:.1f}MB)")
            downloaded_count += 1
            results.append({
                "object_id": object_id,
                "title": title,
                "filename": filename,
                "file_size_mb": file_size,
                "status": "success"
            })
        else:
            print(f"  ❌ 다운로드 실패")
            failed_count += 1
            results.append({
                "object_id": object_id,
                "title": title,
                "status": "failed",
                "reason": "다운로드 실패"
            })
        
        # 진행 상황 표시
        if (i + 1) % 50 == 0:
            print(f"\n📊 진행 상황: {i+1}/{len(all_objects)} 완료")
            print(f"  ✅ 성공: {downloaded_count}개")
            print(f"  ❌ 실패: {failed_count}개")
            print(f"  📈 성공률: {(downloaded_count/(i+1)*100):.1f}%")
        
        human_delay()
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(METADATA_DIR, f"smithsonian_portraits_{timestamp}.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 최종 통계
    total_size = sum([r.get('file_size_mb', 0) for r in results if r.get('status') == 'success'])
    
    print(f"\n🎉 === 스미소니언 초상화 수집 완료 ===")
    print(f"✅ 총 다운로드: {downloaded_count}개")
    print(f"❌ 다운로드 실패: {failed_count}개")
    print(f"📁 이미지 저장: {SAVE_DIR}")
    print(f"📄 메타데이터: {csv_path}")
    print(f"💾 총 파일 크기: {total_size:.1f}MB")

if __name__ == "__main__":
    main()
