import os
import requests
import pandas as pd
import time
from urllib.parse import quote
import json

# 설정
CSV_FILE = "portraits_dataset.csv"
SAVE_DIR = "data/portrait_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 메트로폴리탄 미술관 API 설정
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

def search_met_portraits(title, limit=5):
    """메트로폴리탄 미술관에서 초상화 검색"""
    try:
        # 제목으로 검색
        search_url = f"{MET_API_BASE}/search"
        params = {
            'q': title,
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
                    classification = obj_data.get('classification', '').lower()
                    object_name = obj_data.get('objectName', '').lower()
                    
                    if any(keyword in title_lower or keyword in classification or keyword in object_name 
                           for keyword in ['portrait', 'portraiture', 'self-portrait']):
                        
                        # 이미지 URL 찾기
                        primary_image = obj_data.get('primaryImage')
                        if primary_image:
                            portraits.append({
                                'title': obj_data.get('title', ''),
                                'artist': obj_data.get('artistDisplayName', ''),
                                'image_url': primary_image,
                                'object_id': obj_id,
                                'culture': obj_data.get('culture', ''),
                                'period': obj_data.get('period', ''),
                                'medium': obj_data.get('medium', '')
                            })
                
                time.sleep(0.5)  # API 호출 제한
            
            return portraits
    except Exception as e:
        print(f"검색 오류 ({title}): {e}")
        return []

def download_image(url, filename):
    """이미지 다운로드"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; PortraitDownloader/1.0)'
        }
        response = requests.get(url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            filepath = os.path.join(SAVE_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        else:
            print(f"다운로드 실패: {url} (상태코드: {response.status_code})")
            return None
    except Exception as e:
        print(f"다운로드 오류: {e}")
        return None

def main():
    # CSV 파일 읽기
    df = pd.read_csv(CSV_FILE)
    print(f"총 {len(df)}개의 초상화 정보를 처리합니다.")
    
    downloaded_count = 0
    failed_count = 0
    results = []
    
    # 각 초상화에 대해 검색 및 다운로드
    for idx, row in df.iterrows():
        title = row.get('Title', '')
        print(f"\n[{idx+1}/{len(df)}] 처리 중: {title}")
        
        # 메트로폴리탄 미술관에서 검색
        portraits = search_met_portraits(title)
        
        if portraits:
            # 첫 번째 매칭되는 초상화 다운로드
            portrait = portraits[0]
            
            # 파일명 생성
            artist_name = portrait['artist'].replace('/', '_').replace('\\', '_')[:30]
            title_name = portrait['title'].replace('/', '_').replace('\\', '_')[:30]
            filename = f"{idx:04d}_{artist_name}_{title_name}.jpg"
            
            # 이미지 다운로드
            filepath = download_image(portrait['image_url'], filename)
            
            if filepath:
                print(f"✓ 다운로드 완료: {filename}")
                downloaded_count += 1
                results.append({
                    'original_title': title,
                    'found_title': portrait['title'],
                    'artist': portrait['artist'],
                    'filepath': filepath,
                    'object_id': portrait['object_id'],
                    'culture': portrait['culture'],
                    'period': portrait['period'],
                    'medium': portrait['medium']
                })
            else:
                print(f"✗ 다운로드 실패: {title}")
                failed_count += 1
        else:
            print(f"✗ 매칭되는 초상화를 찾을 수 없음: {title}")
            failed_count += 1
        
        # API 호출 제한을 위한 대기
        time.sleep(1)
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("downloaded_portraits_results.csv", index=False, encoding='utf-8')
    
    print(f"\n=== 다운로드 완료 ===")
    print(f"성공: {downloaded_count}개")
    print(f"실패: {failed_count}개")
    print(f"결과 파일: downloaded_portraits_results.csv")
    print(f"이미지 저장 위치: {SAVE_DIR}")

if __name__ == "__main__":
    main()
