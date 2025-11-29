import os
import requests
import pandas as pd
import time
import json

# 설정
SAVE_DIR = "data/test_portraits"
METADATA_DIR = "data/test_metadata"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# 메트로폴리탄 미술관 API 설정
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

def search_portraits_test(limit=10):
    """테스트용: 제한된 수의 초상화 작품 검색"""
    
    print(f"🔍 테스트: 처음 {limit}개 초상화 작품을 검색합니다...")
    
    try:
        # 'portrait' 키워드로 검색
        search_url = f"{MET_API_BASE}/search"
        params = {
            'q': 'portrait',
            'hasImages': 'true',
            'isHighlight': 'false'
        }
        
        response = requests.get(search_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            object_ids = data.get('objectIDs', [])[:limit]
            print(f"  ✅ {len(object_ids)}개 작품 ID 수집 완료")
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
        response = requests.get(detail_url, timeout=10)
        
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
    ]
    
    combined_text = ' '.join(text_fields).lower()
    
    portrait_keywords = [
        'portrait', 'portraiture', 'self-portrait', 
        'bust', 'head', 'figure', 'profile'
    ]
    
    return any(keyword in combined_text for keyword in portrait_keywords)

def download_portraits_test():
    """테스트용: 제한된 수의 초상화 다운로드"""
    
    print("🎨 테스트: 메트로폴리탄 미술관 초상화 다운로드")
    print("=" * 50)
    
    # 1단계: 초상화 작품 ID 수집 (테스트: 100개)
    object_ids = search_portraits_test(limit=100)
    
    if not object_ids:
        print("❌ 검색된 작품이 없습니다.")
        return
    
    # 2단계: 각 작품 처리
    downloaded_count = 0
    failed_count = 0
    not_portrait_count = 0
    results = []
    
    print(f"\n📥 {len(object_ids)}개 작품을 처리합니다...")
    
    for i, object_id in enumerate(object_ids):
        print(f"\n[{i+1}/{len(object_ids)}] 처리 중: ID {object_id}")
        
        # 상세 정보 가져오기
        obj_data = get_object_details(object_id)
        
        if not obj_data:
            print(f"  ❌ 상세 정보를 가져올 수 없음")
            failed_count += 1
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
        title_clean = re.sub(r'[^\w\s\-]', '', title)[:30].replace(' ', '_')
        artist_clean = re.sub(r'[^\w\s\-]', '', artist)[:20].replace(' ', '_')
        filename = f"{object_id}_{artist_clean}_{title_clean}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # 이미지 다운로드
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; PortraitDownloader/1.0)',
                'Accept': 'image/*'
            }
            response = requests.get(image_url, headers=headers, timeout=20)
            
            if response.status_code == 200 and response.content:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                file_size_mb = len(response.content) / (1024 * 1024)
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
                print(f"  ❌ 다운로드 실패: HTTP {response.status_code}")
                failed_count += 1
                
        except Exception as e:
            print(f"  ❌ 다운로드 오류: {e}")
            failed_count += 1
        
        # API 호출 제한
        time.sleep(0.5)
        
        # 진행 상황 표시 (매 10개마다)
        if (i + 1) % 10 == 0:
            print(f"\n📊 진행 상황: {i+1}/{len(object_ids)} 완료")
            print(f"  ✅ 성공: {downloaded_count}개")
            print(f"  ❌ 실패: {failed_count}개")
            print(f"  ⏭️ 초상화 아님: {not_portrait_count}개")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(METADATA_DIR, "test_portraits_results.csv"), index=False, encoding='utf-8')
    
    print(f"\n🎉 === 테스트 다운로드 완료 ===")
    print(f"✅ 다운로드 성공: {downloaded_count}개")
    print(f"❌ 다운로드 실패: {failed_count}개")
    print(f"⏭️ 초상화 아님: {not_portrait_count}개")
    print(f"📊 총 처리: {len(object_ids)}개")
    print(f"📁 이미지 저장: {SAVE_DIR}")
    print(f"📄 메타데이터: {METADATA_DIR}/test_portraits_results.csv")
    
    return results_df

def analyze_test_results(results_df):
    """테스트 결과 분석"""
    
    if len(results_df) == 0:
        print("분석할 결과가 없습니다.")
        return
    
    print(f"\n📊 === 테스트 결과 분석 ===")
    
    # 미술가별 통계
    artist_counts = results_df['artist'].value_counts()
    print(f"\n👨‍🎨 미술가별 작품 수:")
    for artist, count in artist_counts.head(5).items():
        print(f"  • {artist}: {count}개")
    
    # 시대별 통계
    period_counts = results_df['period'].value_counts()
    print(f"\n⏰ 시대별 작품 수 (상위 5개):")
    for period, count in period_counts.head(5).items():
        if period:  # 빈 값 제외
            print(f"  • {period}: {count}개")
    
    # 분류별 통계
    classification_counts = results_df['classification'].value_counts()
    print(f"\n📂 분류별 작품 수:")
    for classification, count in classification_counts.items():
        if classification:  # 빈 값 제외
            print(f"  • {classification}: {count}개")
    
    # 파일 크기 통계
    total_size_mb = results_df['file_size_mb'].sum()
    avg_size_mb = results_df['file_size_mb'].mean()
    print(f"\n💾 파일 크기 통계:")
    print(f"  • 총 크기: {total_size_mb:.2f}MB")
    print(f"  • 평균 크기: {avg_size_mb:.2f}MB")
    print(f"  • 최대 크기: {results_df['file_size_mb'].max():.2f}MB")
    print(f"  • 최소 크기: {results_df['file_size_mb'].min():.2f}MB")

def main():
    print("🧪 테스트: 메트로폴리탄 미술관 초상화 다운로더")
    print("처음 10개 작품으로 테스트합니다!")
    print("=" * 50)
    
    # 테스트 다운로드 실행
    results_df = download_portraits_test()
    
    # 결과 분석
    if len(results_df) > 0:
        analyze_test_results(results_df)
        
        print(f"\n💡 테스트 결과:")
        print(f"• 성공적으로 다운로드된 초상화들을 확인하세요")
        print(f"• 품질과 정확성이 만족스럽면 전체 버전을 실행하세요")
        print(f"• 다운로드된 이미지들을 AI 분석에 사용할 수 있습니다")

if __name__ == "__main__":
    main()
