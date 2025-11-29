import os
import requests
import pandas as pd
import time
import json
import random
from urllib.parse import urljoin

# 설정
SAVE_DIR = "data/stealth_portraits"
METADATA_DIR = "data/stealth_metadata"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# 메트로폴리탄 미술관 API 설정
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

# 더 인간적인 User-Agent들
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
]

def get_random_user_agent():
    """랜덤 User-Agent 선택"""
    return random.choice(USER_AGENTS)

def human_like_delay():
    """인간적인 지연 시간"""
    # 2-5초 사이의 랜덤 지연
    delay = random.uniform(2.0, 5.0)
    time.sleep(delay)

def search_portraits_stealth(limit=1000):
    """스텔스 모드로 초상화 검색"""
    
    print(f"🔍 스텔스 모드: 최대 {limit:,}개의 초상화 작품을 검색합니다...")
    
    # 초상화 관련 키워드들로 검색
    portrait_keywords = [
        'portrait',
        'self-portrait', 
        'portraiture',
        'bust',
        'head',
        'figure'
    ]
    
    all_object_ids = set()
    
    for keyword in portrait_keywords:
        print(f"  📝 '{keyword}' 검색 중...")
        
        try:
            search_url = f"{MET_API_BASE}/search"
            params = {
                'q': keyword,
                'hasImages': 'true',
                'isHighlight': 'false'
            }
            
            headers = {'User-Agent': get_random_user_agent()}
            response = requests.get(search_url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                object_ids = data.get('objectIDs', [])[:limit]
                all_object_ids.update(object_ids)
                print(f"    ✅ {len(object_ids)}개 작품 발견")
            else:
                print(f"    ❌ 검색 실패: {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ 오류: {e}")
        
        # 키워드 간 지연
        human_like_delay()
    
    print(f"🎯 총 {len(all_object_ids)}개의 고유한 작품 ID를 찾았습니다.")
    return list(all_object_ids)

def get_object_details_stealth(object_id):
    """스텔스 모드로 작품 상세 정보 가져오기"""
    try:
        detail_url = f"{MET_API_BASE}/objects/{object_id}"
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(detail_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"  ❌ 상세 정보 오류 ({object_id}): {e}")
        return None

def is_portrait_work(obj_data):
    """초상화 작품인지 판단"""
    if not obj_data:
        return False
    
    # 제목, 분류, 부서명 등에서 초상화 키워드 확인
    text_fields = [
        obj_data.get('title', ''),
        obj_data.get('classification', ''),
        obj_data.get('objectName', ''),
        obj_data.get('department', ''),
        obj_data.get('culture', ''),
        obj_data.get('period', '')
    ]
    
    combined_text = ' '.join(text_fields).lower()
    
    portrait_keywords = [
        'portrait', 'portraiture', 'self-portrait', 
        'bust', 'head', 'figure', 'profile'
    ]
    
    return any(keyword in combined_text for keyword in portrait_keywords)

def download_image_stealth(url, filename):
    """스텔스 모드로 이미지 다운로드"""
    try:
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'image/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            filepath = os.path.join(SAVE_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        else:
            print(f"  ❌ 다운로드 실패: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ 다운로드 오류: {e}")
        return None

def download_portraits_stealth():
    """스텔스 모드 대량 초상화 다운로드"""
    
    print("🥷 스텔스 모드: 메트로폴리탄 미술관 초상화 다운로더")
    print("인간적인 속도와 패턴으로 봇 탐지를 우회합니다!")
    print("=" * 60)
    
    # 1단계: 초상화 작품 ID 수집
    all_object_ids = search_portraits_stealth(limit=1000)
    
    if not all_object_ids:
        print("❌ 검색된 작품이 없습니다.")
        return
    
    # 2단계: 각 작품의 상세 정보 확인 및 다운로드
    downloaded_count = 0
    failed_count = 0
    not_portrait_count = 0
    api_error_count = 0
    results = []
    
    print(f"\n📥 {len(all_object_ids)}개 작품을 스텔스 모드로 처리합니다...")
    
    for i, object_id in enumerate(all_object_ids):
        print(f"\n[{i+1}/{len(all_object_ids)}] 처리 중: ID {object_id}")
        
        # 상세 정보 가져오기
        obj_data = get_object_details_stealth(object_id)
        
        if not obj_data:
            print(f"  ❌ 상세 정보를 가져올 수 없음")
            api_error_count += 1
            # API 오류 시 더 긴 지연
            time.sleep(10)
            continue
        
        # 초상화 작품인지 확인
        if not is_portrait_work(obj_data):
            print(f"  ⏭️ 초상화가 아님 - 건너뜀")
            not_portrait_count += 1
            continue
        
        # 이미지 URL 확인
        image_url = obj_data.get('primaryImage')
        if not image_url or not image_url.strip():
            print(f"  ❌ 이미지 URL 없음")
            failed_count += 1
            continue
        
        # 작품 정보 출력
        title = obj_data.get('title', 'Untitled')
        artist = obj_data.get('artistDisplayName', 'Unknown')
        print(f"  🎨 {artist} - {title}")
        
        # 파일명 생성
        import re
        title_clean = re.sub(r'[^\w\s\-]', '', title)[:50].replace(' ', '_')
        artist_clean = re.sub(r'[^\w\s\-]', '', artist)[:30].replace(' ', '_')
        filename = f"{object_id}_{artist_clean}_{title_clean}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # 이미지 다운로드
        filepath = download_image_stealth(image_url, filename)
        
        if filepath:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ 다운로드 완료: {filename} ({file_size_mb:.1f}MB)")
            
            downloaded_count += 1
            results.append({
                'object_id': object_id,
                'title': title,
                'artist': artist,
                'image_url': image_url,
                'filename': filename,
                'filepath': filepath,
                'file_size_mb': file_size_mb,
                'classification': obj_data.get('classification', ''),
                'department': obj_data.get('department', ''),
                'culture': obj_data.get('culture', ''),
                'period': obj_data.get('period', ''),
                'medium': obj_data.get('medium', ''),
                'object_date': obj_data.get('objectDate', ''),
                'status': 'success'
            })
        else:
            print(f"  ❌ 다운로드 실패: {title}")
            failed_count += 1
        
        # 인간적인 지연 시간
        human_like_delay()
        
        # 진행 상황 표시 (매 50개마다)
        if (i + 1) % 50 == 0:
            print(f"\n📊 진행 상황: {i+1:,}/{len(all_object_ids):,} 완료")
            print(f"  ✅ 성공: {downloaded_count:,}개")
            print(f"  ❌ 실패: {failed_count:,}개")
            print(f"  ⏭️ 초상화 아님: {not_portrait_count:,}개")
            print(f"  🚫 API 오류: {api_error_count:,}개")
            print(f"  📈 성공률: {(downloaded_count/(i+1)*100):.1f}%")
            
            # API 오류가 많으면 더 긴 휴식
            if api_error_count > 10:
                print(f"  😴 API 오류가 많아 30초 휴식...")
                time.sleep(30)
                api_error_count = 0  # 카운터 리셋
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(METADATA_DIR, "stealth_portraits_results.csv"), index=False, encoding='utf-8')
    
    print(f"\n🎉 === 스텔스 모드 다운로드 완료 ===")
    print(f"✅ 다운로드 성공: {downloaded_count}개")
    print(f"❌ 다운로드 실패: {failed_count}개")
    print(f"⏭️ 초상화 아님: {not_portrait_count}개")
    print(f"🚫 API 오류: {api_error_count}개")
    print(f"📊 총 처리: {len(all_object_ids)}개")
    print(f"📁 이미지 저장: {SAVE_DIR}")
    print(f"📄 메타데이터: {METADATA_DIR}/stealth_portraits_results.csv")
    
    return results_df

def main():
    print("🥷 스텔스 모드 초상화 다운로더")
    print("봇 탐지를 우회하여 안전하게 다운로드합니다!")
    print("=" * 60)
    
    # 스텔스 모드 다운로드 실행
    results_df = download_portraits_stealth()
    
    if len(results_df) > 0:
        print(f"\n💡 스텔스 모드 결과:")
        print(f"• 성공적으로 다운로드된 초상화들을 확인하세요")
        print(f"• 봇 탐지를 우회하여 안전하게 수집했습니다")
        print(f"• 다운로드된 이미지들을 AI 분석에 사용할 수 있습니다")

if __name__ == "__main__":
    main()
