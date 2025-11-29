import os
import requests
import pandas as pd
import time
import re
from urllib.parse import quote

# 설정
CSV_FILE = "portraits_dataset.csv"
SAVE_DIR = "data/csv_portraits_improved"
os.makedirs(SAVE_DIR, exist_ok=True)

# 메트로폴리탄 미술관 API 설정
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

def create_detailed_search_query(title, medium, classification):
    """제목, 재료, 분류를 모두 포함한 상세 검색 쿼리 생성"""
    
    # 검색 쿼리 조합
    search_terms = []
    
    # 제목 추가
    if title:
        search_terms.append(title)
    
    # 재료에서 키워드 추출
    if medium:
        medium_keywords = []
        if 'oil' in medium.lower():
            medium_keywords.append('oil')
        if 'canvas' in medium.lower():
            medium_keywords.append('canvas')
        if 'panel' in medium.lower():
            medium_keywords.append('panel')
        if 'paper' in medium.lower():
            medium_keywords.append('paper')
        if 'bronze' in medium.lower():
            medium_keywords.append('bronze')
        if 'etching' in medium.lower():
            medium_keywords.append('etching')
        if 'lithograph' in medium.lower():
            medium_keywords.append('lithograph')
        if 'photograph' in medium.lower():
            medium_keywords.append('photograph')
        
        search_terms.extend(medium_keywords)
    
    # 분류 추가
    if classification:
        search_terms.append(classification.lower())
    
    # 중복 제거하고 조합
    unique_terms = list(set(search_terms))
    return ' '.join(unique_terms)

def search_met_portrait_detailed(title, medium, classification, limit=3):
    """상세 검색으로 메트로폴리탄 미술관에서 초상화 검색"""
    try:
        # 상세 검색 쿼리 생성
        search_query = create_detailed_search_query(title, medium, classification)
        print(f"  🔍 검색 쿼리: '{search_query}'")
        
        # 제목으로 검색
        search_url = f"{MET_API_BASE}/search"
        params = {
            'q': search_query,
            'hasImages': 'true',
            'isHighlight': 'false'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            object_ids = data.get('objectIDs', [])[:limit]
            
            portraits = []
            for obj_id in object_ids:
                # 각 작품의 상세 정보 가져오기
                detail_url = f"{MET_API_BASE}/objects/{obj_id}"
                detail_response = requests.get(detail_url, timeout=10)
                
                if detail_response.status_code == 200:
                    obj_data = detail_response.json()
                    
                    # 초상화 관련 키워드 확인
                    title_lower = obj_data.get('title', '').lower()
                    classification_lower = obj_data.get('classification', '').lower()
                    object_name = obj_data.get('objectName', '').lower()
                    department = obj_data.get('department', '').lower()
                    
                    # 초상화 관련 키워드들
                    portrait_keywords = ['portrait', 'portraiture', 'self-portrait', 'bust', 'head', 'figure']
                    
                    if any(keyword in title_lower or keyword in classification_lower or keyword in object_name 
                           for keyword in portrait_keywords):
                        
                        # 이미지 URL 찾기
                        primary_image = obj_data.get('primaryImage')
                        if primary_image and primary_image.strip():
                            portraits.append({
                                'title': obj_data.get('title', ''),
                                'artist': obj_data.get('artistDisplayName', ''),
                                'image_url': primary_image,
                                'object_id': obj_id,
                                'culture': obj_data.get('culture', ''),
                                'period': obj_data.get('period', ''),
                                'medium': obj_data.get('medium', ''),
                                'department': obj_data.get('department', ''),
                                'object_date': obj_data.get('objectDate', ''),
                                'classification': obj_data.get('classification', '')
                            })
                
                time.sleep(0.3)  # API 호출 제한
            
            return portraits
    except Exception as e:
        print(f"  ❌ 검색 오류 ({search_query}): {e}")
        return []

def download_image(url, filename):
    """이미지 다운로드"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; PortraitDownloader/1.0)',
            'Accept': 'image/*'
        }
        response = requests.get(url, headers=headers, timeout=20)
        
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

def process_csv_portraits_improved():
    """개선된 방식으로 CSV 파일의 초상화들을 처리합니다."""
    
    # CSV 파일 읽기
    df = pd.read_csv(CSV_FILE)
    print(f"📋 CSV 파일에서 {len(df)}개의 초상화 정보를 로드했습니다.")
    
    downloaded_count = 0
    failed_count = 0
    no_match_count = 0
    duplicate_count = 0
    results = []
    
    # 처음 20개만 테스트로 처리 (전체 처리하려면 이 줄을 제거하세요)
    df_sample = df.head(20).copy()
    print(f"🔍 처음 {len(df_sample)}개 작품을 테스트로 처리합니다.")
    
    # 이미 다운로드된 이미지 URL들을 추적 (중복 방지)
    downloaded_urls = set()
    
    for idx, row in df_sample.iterrows():
        title = row.get('Title', '').strip()
        medium = row.get('Medium', '').strip()
        classification = row.get('Classification', '').strip()
        
        if not title:
            continue
            
        print(f"\n[{idx+1}/{len(df_sample)}] 처리 중: '{title}'")
        print(f"  📝 재료: {medium}")
        print(f"  📂 분류: {classification}")
        
        # 상세 검색
        portraits = search_met_portrait_detailed(title, medium, classification)
        
        if portraits:
            # 첫 번째 매칭되는 초상화 선택
            portrait = portraits[0]
            
            # 중복 URL 체크
            if portrait['image_url'] in downloaded_urls:
                print(f"  ⚠️ 중복 이미지 발견 - 건너뜀")
                duplicate_count += 1
                results.append({
                    'csv_index': idx,
                    'csv_title': title,
                    'csv_medium': medium,
                    'csv_classification': classification,
                    'found_title': portrait['title'],
                    'found_artist': portrait['artist'],
                    'filepath': '',
                    'status': 'duplicate_skipped'
                })
                continue
            
            # 파일명 생성
            artist_name = re.sub(r"[^\w\s\-]", "", portrait['artist'])[:25].replace(" ", "_")
            title_name = re.sub(r"[^\w\s\-]", "", portrait['title'])[:25].replace(" ", "_")
            medium_short = re.sub(r"[^\w\s\-]", "", medium)[:15].replace(" ", "_")
            filename = f"{idx:04d}_{artist_name}_{title_name}_{medium_short}.jpg"
            
            # 이미지 다운로드
            filepath = download_image(portrait['image_url'], filename)
            
            if filepath:
                print(f"  ✅ 다운로드 완료: {filename}")
                downloaded_count += 1
                downloaded_urls.add(portrait['image_url'])  # URL 기록
                results.append({
                    'csv_index': idx,
                    'csv_title': title,
                    'csv_medium': medium,
                    'csv_classification': classification,
                    'found_title': portrait['title'],
                    'found_artist': portrait['artist'],
                    'filepath': filepath,
                    'object_id': portrait['object_id'],
                    'culture': portrait['culture'],
                    'period': portrait['period'],
                    'medium_found': portrait['medium'],
                    'department': portrait['department'],
                    'classification_found': portrait['classification'],
                    'status': 'success'
                })
            else:
                print(f"  ❌ 다운로드 실패: {title}")
                failed_count += 1
        else:
            print(f"  🔍 매칭되는 초상화를 찾을 수 없음: {title}")
            no_match_count += 1
            results.append({
                'csv_index': idx,
                'csv_title': title,
                'csv_medium': medium,
                'csv_classification': classification,
                'found_title': '',
                'found_artist': '',
                'filepath': '',
                'status': 'no_match'
            })
        
        # API 호출 제한을 위한 대기
        time.sleep(1)
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("csv_portraits_improved_results.csv", index=False, encoding='utf-8')
    
    print(f"\n🎉 === 개선된 처리 완료 ===")
    print(f"✅ 다운로드 성공: {downloaded_count}개")
    print(f"❌ 다운로드 실패: {failed_count}개")
    print(f"🔍 매칭 없음: {no_match_count}개")
    print(f"⚠️ 중복 건너뜀: {duplicate_count}개")
    print(f"📊 총 처리: {len(df_sample)}개")
    print(f"📁 결과 파일: csv_portraits_improved_results.csv")
    print(f"🖼️ 이미지 저장 위치: {SAVE_DIR}")
    
    return results_df

def analyze_improved_results(results_df):
    """개선된 결과를 분석합니다."""
    
    if len(results_df) == 0:
        print("분석할 결과가 없습니다.")
        return
    
    print(f"\n📊 === 개선된 결과 분석 ===")
    
    # 상태별 통계
    status_counts = results_df['status'].value_counts()
    print(f"\n📈 상태별 통계:")
    for status, count in status_counts.items():
        print(f"  • {status}: {count}개")
    
    # 성공한 작품들의 정보
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(f"\n🎨 성공한 작품들:")
        for _, row in successful.iterrows():
            print(f"  • {row['found_artist']} - {row['found_title']}")
            print(f"    CSV: {row['csv_title']} ({row['csv_medium']})")
            print(f"    Found: {row['found_title']} ({row['medium_found']})")
            print()
        
        # 재료별 통계
        medium_counts = successful['csv_medium'].value_counts()
        print(f"\n🎨 CSV 재료별 작품 수:")
        for medium, count in medium_counts.head(10).items():
            print(f"  • {medium}: {count}개")

def main():
    print("🎨 개선된 CSV 초상화 이미지 다운로더")
    print("제목 + 재료 + 분류를 모두 고려한 상세 검색으로 중복을 방지합니다!")
    print("=" * 70)
    
    # CSV 처리
    results_df = process_csv_portraits_improved()
    
    # 결과 분석
    analyze_improved_results(results_df)
    
    print(f"\n💡 개선 사항:")
    print(f"• 제목 + 재료 + 분류를 모두 포함한 상세 검색")
    print(f"• 중복 URL 자동 감지 및 건너뛰기")
    print(f"• 파일명에 재료 정보 포함")
    print(f"• 더 정확한 작품 매칭")

if __name__ == "__main__":
    main()
