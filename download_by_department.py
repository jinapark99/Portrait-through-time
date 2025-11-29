import os
import requests
import pandas as pd
import time
import json
import random

# 설정
SAVE_DIR = "data/department_portraits"
METADATA_DIR = "data/department_metadata"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# 메트로폴리탄 미술관 API 설정
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

# User-Agent 리스트
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def human_delay():
    """인간적인 지연"""
    delay = random.uniform(3.0, 6.0)
    time.sleep(delay)

def get_department_objects(department_id, limit=5000):
    """특정 부서의 모든 작품 ID 가져오기"""
    
    print(f"🏛️ {department_id} 부서의 작품들을 검색합니다...")
    
    try:
        # 부서별 검색
        search_url = f"{MET_API_BASE}/search"
        params = {
            'departmentId': department_id,
            'hasImages': 'true',
            'isHighlight': 'false'
        }
        
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(search_url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            object_ids = data.get('objectIDs', [])[:limit]
            print(f"  ✅ {len(object_ids)}개 작품 발견")
            return object_ids
        else:
            print(f"  ❌ 검색 실패: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return []

def get_object_details(object_id):
    """작품 상세 정보 가져오기"""
    try:
        detail_url = f"{MET_API_BASE}/objects/{object_id}"
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(detail_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

def is_portrait_advanced(obj_data):
    """고급 초상화 판단 로직"""
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
    
    combined_text = ' '.join(text_fields).lower()
    
    # 포괄적인 초상화 키워드들
    portrait_keywords = [
        # 기본 키워드
        'portrait', 'portraiture', 'self-portrait', 'self portrait',
        'bust', 'head', 'figure', 'profile',
        'likeness', 'representation', 'image of',
        
        # 구체적인 표현들
        'portrait of', 'portrait of a', 'portrait of the',
        'head of', 'bust of', 'figure of',
        
        # 인물 관련
        'man', 'woman', 'lady', 'gentleman', 'person', 'people',
        'child', 'boy', 'girl', 'infant', 'baby',
        'elderly', 'old man', 'old woman',
        
        # 직업/신분
        'king', 'queen', 'prince', 'princess', 'emperor', 'empress',
        'noble', 'nobleman', 'noblewoman', 'aristocrat',
        'monk', 'nun', 'priest', 'bishop', 'cardinal',
        'merchant', 'banker', 'scholar', 'artist', 'painter',
        
        # 자화상 관련
        'autoportrait', 'autoritratto', 'selbstportrait',
        
        # 기타 표현들
        'effigy', 'statue', 'sculpture of', 'painting of'
    ]
    
    # 키워드 매칭 확인
    keyword_matches = sum(1 for keyword in portrait_keywords if keyword in combined_text)
    
    # 최소 1개 이상의 키워드 매칭 필요
    if keyword_matches == 0:
        return False
    
    # 제외할 키워드들 (초상화가 아닌 경우)
    exclude_keywords = [
        'landscape', 'still life', 'still-life', 'nature', 'flower',
        'animal', 'architecture', 'building', 'interior', 'exterior',
        'mythology', 'mythological', 'allegory', 'allegorical',
        'battle', 'war', 'scene', 'event', 'story'
    ]
    
    # 제외 키워드가 많이 포함되면 제외
    exclude_matches = sum(1 for keyword in exclude_keywords if keyword in combined_text)
    
    # 제외 키워드가 많으면 초상화가 아닐 가능성이 높음
    if exclude_matches > 2:
        return False
    
    # 최종 판단: 키워드 매칭이 있고 제외 키워드가 적으면 초상화
    return keyword_matches > 0 and exclude_matches <= 2

def download_image(url, filename):
    """이미지 다운로드"""
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
            return None
    except Exception as e:
        return None

def scan_department_for_portraits(department_name, department_id, limit=1000):
    """부서별 초상화 스캔"""
    
    print(f"\n🏛️ === {department_name} 부서 초상화 스캔 ===")
    
    # 1단계: 부서의 모든 작품 ID 가져오기
    object_ids = get_department_objects(department_id, limit)
    
    if not object_ids:
        print(f"❌ {department_name} 부서에서 작품을 찾을 수 없습니다.")
        return []
    
    # 2단계: 각 작품 분석 및 초상화 다운로드
    downloaded_count = 0
    failed_count = 0
    not_portrait_count = 0
    api_error_count = 0
    results = []
    
    print(f"\n📥 {len(object_ids)}개 작품을 분석합니다...")
    
    for i, object_id in enumerate(object_ids):
        print(f"\n[{i+1}/{len(object_ids)}] 분석 중: ID {object_id}")
        
        # 상세 정보 가져오기
        obj_data = get_object_details(object_id)
        
        if not obj_data:
            print(f"  ❌ 상세 정보를 가져올 수 없음")
            api_error_count += 1
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
        filename = f"{department_id}_{object_id}_{artist_clean}_{title_clean}.jpg"
        
        # 이미지 다운로드
        filepath = download_image(image_url, filename)
        
        if filepath:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ 다운로드 완료: {filename} ({file_size_mb:.1f}MB)")
            
            downloaded_count += 1
            results.append({
                'department_id': department_id,
                'department_name': department_name,
                'object_id': object_id,
                'title': title,
                'artist': artist,
                'image_url': image_url,
                'filename': filename,
                'filepath': filepath,
                'file_size_mb': file_size_mb,
                'classification': obj_data.get('classification', ''),
                'culture': obj_data.get('culture', ''),
                'period': obj_data.get('period', ''),
                'medium': obj_data.get('medium', ''),
                'object_date': obj_data.get('objectDate', ''),
                'status': 'success'
            })
        else:
            print(f"  ❌ 다운로드 실패: {title}")
            failed_count += 1
        
        # 인간적인 지연
        human_delay()
        
        # 진행 상황 표시 (매 50개마다)
        if (i + 1) % 50 == 0:
            print(f"\n📊 진행 상황: {i+1:,}/{len(object_ids):,} 완료")
            print(f"  ✅ 성공: {downloaded_count:,}개")
            print(f"  ❌ 실패: {failed_count:,}개")
            print(f"  ⏭️ 초상화 아님: {not_portrait_count:,}개")
            print(f"  🚫 API 오류: {api_error_count:,}개")
            print(f"  📈 성공률: {(downloaded_count/(i+1)*100):.1f}%")
    
    print(f"\n🎉 === {department_name} 부서 스캔 완료 ===")
    print(f"✅ 다운로드 성공: {downloaded_count}개")
    print(f"❌ 다운로드 실패: {failed_count}개")
    print(f"⏭️ 초상화 아님: {not_portrait_count}개")
    print(f"🚫 API 오류: {api_error_count}개")
    print(f"📊 총 처리: {len(object_ids)}개")
    
    return results

def main():
    print("🏛️ 메트로폴리탄 미술관 부서별 초상화 스캔")
    print("가장 정확도 높은 방법으로 초상화를 수집합니다!")
    print("=" * 60)
    
    # 초상화가 많은 주요 부서들
    departments = [
        {
            'id': 11,  # European Paintings
            'name': 'European Paintings'
        },
        {
            'id': 1,   # American Paintings and Sculpture
            'name': 'American Paintings and Sculpture'
        },
        {
            'id': 21,  # Modern and Contemporary Art
            'name': 'Modern and Contemporary Art'
        },
        {
            'id': 2,   # The American Wing
            'name': 'The American Wing'
        }
    ]
    
    all_results = []
    
    for dept in departments:
        # 각 부서별로 스캔
        results = scan_department_for_portraits(
            dept['name'], 
            dept['id'], 
            limit=500  # 각 부서당 500개씩 테스트
        )
        all_results.extend(results)
        
        # 부서 간 지연
        print(f"\n😴 다음 부서로 넘어가기 전 60초 휴식...")
        time.sleep(60)
    
    # 전체 결과 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(METADATA_DIR, "department_portraits_results.csv"), index=False, encoding='utf-8')
        
        print(f"\n🎉 === 전체 부서별 스캔 완료 ===")
        print(f"✅ 총 다운로드: {len(all_results)}개")
        print(f"📁 이미지 저장: {SAVE_DIR}")
        print(f"📄 메타데이터: {METADATA_DIR}/department_portraits_results.csv")
        
        # 부서별 통계
        dept_stats = results_df.groupby('department_name').size()
        print(f"\n📊 부서별 작품 수:")
        for dept, count in dept_stats.items():
            print(f"  • {dept}: {count}개")
    else:
        print(f"\n❌ 다운로드된 초상화가 없습니다.")

if __name__ == "__main__":
    main()
