import os
import requests
import pandas as pd
import time
import json
import random

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
    """인간적인 지연 (빠른 버전)"""
    delay = random.uniform(1.5, 3.0)
    time.sleep(delay)

def search_portraits_by_keywords(keywords, limit=2000):
    """다양한 키워드로 초상화 검색"""
    
    print(f"🔍 키워드로 초상화 검색: {', '.join(keywords[:5])}...")
    
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
            response = requests.get(search_url, params=params, headers=headers, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                object_ids = data.get('objectIDs', [])
                
                if object_ids:
                    all_object_ids.update(object_ids[:limit])
                    print(f"    ✅ {len(object_ids)} 개 발견 (총 {len(all_object_ids)} 개 수집)")
                else:
                    print(f"    ⏭️ 결과 없음")
            elif response.status_code == 502:
                print(f"    ⚠️ 502 에러 - 30초 대기 후 재시도...")
                time.sleep(30)
                continue
            else:
                print(f"    ❌ 검색 실패: {response.status_code}")
            
            # 키워드 간 지연 (빠른 버전)
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception as e:
            print(f"    ❌ 오류: {e}")
            continue
    
    print(f"\n✅ 총 {len(all_object_ids)} 개의 고유한 작품 발견!")
    return list(all_object_ids)

def get_object_details_with_retry(object_id, max_retries=3):
    """재시도 로직을 포함한 작품 상세 정보 가져오기"""
    
    for attempt in range(max_retries):
        try:
            detail_url = f"{MET_API_BASE}/objects/{object_id}"
            headers = {'User-Agent': get_random_user_agent()}
            response = requests.get(detail_url, headers=headers, timeout=20)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 502:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"    ⚠️ 502 에러 - {wait_time}초 대기 후 재시도 ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ 502 에러 - 최대 재시도 횟수 초과")
                    return None
            else:
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️ 오류: {e} - 재시도 중...")
                time.sleep(10)
            else:
                return None
    
    return None

def is_portrait_advanced(obj_data):
    """고급 초상화 판단 로직 (더욱 정확하게)"""
    if not obj_data:
        return False
    
    # 텍스트 필드들 수집
    text_fields = [
        obj_data.get('title', ''),
        obj_data.get('classification', ''),
        obj_data.get('objectName', ''),
        obj_data.get('department', ''),
        obj_data.get('culture', ''),
        obj_data.get('period', ''),
        obj_data.get('medium', ''),
        obj_data.get('artistDisplayName', '')
    ]
    
    # tags 처리 (딕셔너리일 수 있음)
    tags = obj_data.get('tags', [])
    if tags and isinstance(tags, list):
        tag_texts = [t.get('term', '') if isinstance(t, dict) else str(t) for t in tags]
        text_fields.append(' '.join(tag_texts))
    
    combined_text = ' '.join([str(f) for f in text_fields]).lower()
    
    # 강력한 초상화 표시자들
    strong_indicators = [
        'portrait', 'portraiture', 'self-portrait', 'self portrait',
        'bust', 'head and shoulders', 'likeness'
    ]
    
    # 약한 초상화 표시자들
    weak_indicators = [
        'man', 'woman', 'lady', 'gentleman', 'person',
        'king', 'queen', 'prince', 'princess',
        'nobleman', 'noblewoman', 'child', 'boy', 'girl'
    ]
    
    # 제외할 키워드들
    exclude_keywords = [
        'landscape', 'still life', 'still-life', 'nature', 'flower', 'flowers',
        'animal', 'animals', 'dog', 'cat', 'horse', 'bird',
        'architecture', 'building', 'church', 'interior', 'exterior',
        'mythology', 'mythological', 'allegory', 'allegorical',
        'battle', 'war scene', 'religious scene', 'biblical scene',
        'crucifixion', 'annunciation', 'nativity'
    ]
    
    # 강력한 표시자 확인
    strong_match = any(indicator in combined_text for indicator in strong_indicators)
    
    # 약한 표시자 확인
    weak_match_count = sum(1 for indicator in weak_indicators if indicator in combined_text)
    
    # 제외 키워드 확인
    exclude_match_count = sum(1 for keyword in exclude_keywords if keyword in combined_text)
    
    # 판단 로직
    if strong_match and exclude_match_count <= 1:
        return True
    
    if weak_match_count >= 2 and exclude_match_count == 0:
        return True
    
    return False

def download_image(url, filename):
    """이미지 다운로드 (재시도 로직 포함)"""
    max_retries = 3
    
    for attempt in range(max_retries):
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
            elif response.status_code == 502 and attempt < max_retries - 1:
                print(f"    ⚠️ 이미지 다운로드 502 에러 - 재시도...")
                time.sleep(15)
            else:
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                return None
    
    return None

def main():
    print("🎨 메트로폴리탄 미술관 스마트 초상화 수집기")
    print("안정적이고 정확한 방법으로 초상화를 수집합니다!")
    print("=" * 60)
    
    # 다양한 초상화 검색 키워드
    search_keywords = [
        # 기본 키워드
        'portrait painting',
        'self-portrait',
        'royal portrait',
        'portrait of a man',
        'portrait of a woman',
        
        # 시대별
        'renaissance portrait',
        'baroque portrait',
        'victorian portrait',
        'eighteenth century portrait',
        'nineteenth century portrait',
        
        # 스타일별
        'oil portrait',
        'portrait drawing',
        'portrait miniature',
        
        # 특정 주제
        'portrait of a lady',
        'portrait of a gentleman',
        'portrait of a child',
        'portrait of an artist',
        'portrait of a nobleman',
        
        # 유명 화가들
        'rembrandt portrait',
        'van gogh portrait',
        'velazquez portrait',
        'holbein portrait',
        'titian portrait'
    ]
    
    # 1단계: 키워드로 작품 검색
    print("\n🔍 1단계: 다양한 키워드로 초상화 검색")
    all_object_ids = search_portraits_by_keywords(search_keywords, limit=5000)
    
    if not all_object_ids:
        print("❌ 검색된 작품이 없습니다. 나중에 다시 시도해주세요.")
        return
    
    print(f"\n✅ {len(all_object_ids)} 개 작품 검색 완료!")
    print(f"📥 2단계: 각 작품 분석 및 초상화 다운로드 시작")
    
    # 2단계: 각 작품 분석 및 다운로드
    downloaded_count = 0
    failed_count = 0
    not_portrait_count = 0
    api_error_count = 0
    results = []
    
    # 처음 1000개만 처리 (테스트)
    object_ids_to_process = all_object_ids[:1000]
    
    for i, object_id in enumerate(object_ids_to_process):
        print(f"\n[{i+1}/{len(object_ids_to_process)}] 분석 중: ID {object_id}")
        
        # 상세 정보 가져오기 (재시도 포함)
        obj_data = get_object_details_with_retry(object_id)
        
        if not obj_data:
            print(f"  ❌ 상세 정보를 가져올 수 없음")
            api_error_count += 1
            
            # API 오류가 많으면 휴식
            if api_error_count > 10:
                print(f"  😴 API 오류가 많아 30초 휴식...")
                time.sleep(30)
                api_error_count = 0
            continue
        
        # 고급 초상화 판단
        if not is_portrait_advanced(obj_data):
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
        
        # 이미지 다운로드
        filepath = download_image(image_url, filename)
        
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
                'object_begin_date': obj_data.get('objectBeginDate', ''),
                'object_end_date': obj_data.get('objectEndDate', ''),
                'dimensions': obj_data.get('dimensions', ''),
                'credit_line': obj_data.get('creditLine', ''),
                'status': 'success'
            })
            
            # CSV 중간 저장 (100개마다)
            if downloaded_count % 100 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(os.path.join(METADATA_DIR, f"portraits_temp_{downloaded_count}.csv"), index=False, encoding='utf-8')
                print(f"  💾 중간 저장: {downloaded_count}개")
        else:
            print(f"  ❌ 다운로드 실패: {title}")
            failed_count += 1
        
        # 인간적인 지연 (더 길게)
        human_delay()
        
        # 진행 상황 표시 (매 50개마다)
        if (i + 1) % 50 == 0:
            print(f"\n📊 진행 상황: {i+1:,}/{len(object_ids_to_process):,} 완료")
            print(f"  ✅ 성공: {downloaded_count:,}개")
            print(f"  ❌ 실패: {failed_count:,}개")
            print(f"  ⏭️ 초상화 아님: {not_portrait_count:,}개")
            print(f"  🚫 API 오류: {api_error_count:,}개")
            if i > 0:
                print(f"  📈 초상화 발견율: {(downloaded_count/(i+1)*100):.1f}%")
            
            if downloaded_count > 0:
                total_size_mb = sum([r.get('file_size_mb', 0) for r in results])
                print(f"  💾 현재 총 크기: {total_size_mb:.1f}MB")
    
    # 최종 결과 저장
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(METADATA_DIR, "smart_portraits_final.csv"), index=False, encoding='utf-8')
        
        print(f"\n🎉 === 스마트 초상화 수집 완료 ===")
        print(f"✅ 총 다운로드: {downloaded_count}개")
        print(f"❌ 다운로드 실패: {failed_count}개")
        print(f"⏭️ 초상화 아님: {not_portrait_count}개")
        print(f"🚫 API 오류: {api_error_count}개")
        print(f"📁 이미지 저장: {SAVE_DIR}")
        print(f"📄 메타데이터: {METADATA_DIR}/smart_portraits_final.csv")
        
        total_size_mb = sum([r.get('file_size_mb', 0) for r in results])
        print(f"💾 총 파일 크기: {total_size_mb:.1f}MB")
    else:
        print(f"\n❌ 다운로드된 초상화가 없습니다.")

if __name__ == "__main__":
    main()

