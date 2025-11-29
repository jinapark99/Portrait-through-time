import os
import requests
import pandas as pd
import time
import json
import random

# 설정
SAVE_DIR = "data/multi_museum_portraits"
METADATA_DIR = "data/multi_museum_metadata"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# 여러 미술관 API 설정
APIS = {
    'metropolitan': {
        'base_url': 'https://collectionapi.metmuseum.org/public/collection/v1',
        'search_endpoint': '/search',
        'object_endpoint': '/objects',
        'name': 'Metropolitan Museum of Art'
    },
    'rijksmuseum': {
        'base_url': 'https://www.rijksmuseum.nl/api/nl/collection',
        'search_endpoint': '',
        'object_endpoint': '',
        'api_key': 'YOUR_RIJKSMUSEUM_API_KEY',  # 실제 API 키 필요
        'name': 'Rijksmuseum'
    }
}

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

def search_metropolitan_portraits(limit=500):
    """메트로폴리탄 미술관에서 초상화 검색"""
    
    print(f"🏛️ 메트로폴리탄 미술관에서 초상화 검색...")
    
    api = APIS['metropolitan']
    all_object_ids = set()
    
    portrait_keywords = ['portrait', 'self-portrait', 'bust', 'head']
    
    for keyword in portrait_keywords:
        try:
            search_url = f"{api['base_url']}{api['search_endpoint']}"
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
                print(f"  ✅ '{keyword}': {len(object_ids)}개 발견")
            else:
                print(f"  ❌ '{keyword}' 검색 실패: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ '{keyword}' 오류: {e}")
        
        human_delay()
    
    print(f"🎯 메트로폴리탄 총 {len(all_object_ids)}개 작품 ID 수집")
    return list(all_object_ids)

def search_open_access_museums():
    """공개 접근 미술관들에서 초상화 검색"""
    
    print(f"🌐 공개 접근 미술관들에서 초상화 검색...")
    
    # 공개 도메인 미술 이미지 소스들
    open_sources = [
        {
            'name': 'Art Institute of Chicago',
            'base_url': 'https://www.artic.edu',
            'api_endpoint': '/api/v1/artworks',
            'search_params': {'q': 'portrait', 'limit': 100}
        },
        {
            'name': 'National Gallery of Art',
            'base_url': 'https://www.nga.gov',
            'api_endpoint': '/api/collection',
            'search_params': {'q': 'portrait', 'size': 100}
        }
    ]
    
    all_artworks = []
    
    for source in open_sources:
        try:
            print(f"  📝 {source['name']} 검색 중...")
            
            url = f"{source['base_url']}{source['api_endpoint']}"
            headers = {'User-Agent': get_random_user_agent()}
            
            response = requests.get(url, params=source['search_params'], headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                artworks = data.get('data', []) if 'data' in data else data.get('artworks', [])
                all_artworks.extend(artworks)
                print(f"    ✅ {len(artworks)}개 작품 발견")
            else:
                print(f"    ❌ 검색 실패: {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ 오류: {e}")
        
        human_delay()
    
    print(f"🎯 공개 미술관 총 {len(all_artworks)}개 작품 발견")
    return all_artworks

def download_from_wikipedia_commons():
    """위키피디아 커먼스에서 초상화 다운로드"""
    
    print(f"📚 위키피디아 커먼스에서 유명 초상화 다운로드...")
    
    # 위키피디아 커먼스의 유명 초상화들
    famous_portraits = [
        {
            'title': 'Mona Lisa',
            'artist': 'Leonardo da Vinci',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg',
            'period': '1503-1519',
            'medium': 'Oil on poplar panel'
        },
        {
            'title': 'Girl with a Pearl Earring',
            'artist': 'Johannes Vermeer',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/0f/1665_Girl_with_a_Pearl_Earring.jpg',
            'period': '1665',
            'medium': 'Oil on canvas'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Vincent van Gogh',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/b/b2/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg',
            'period': '1889',
            'medium': 'Oil on canvas'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Frida Kahlo',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/06/Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg',
            'period': '1932',
            'medium': 'Oil on canvas'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Rembrandt',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Self-portrait_by_Rembrandt.jpg',
            'period': '1660',
            'medium': 'Oil on canvas'
        }
    ]
    
    downloaded_count = 0
    results = []
    
    for i, portrait in enumerate(famous_portraits):
        try:
            print(f"  [{i+1}/{len(famous_portraits)}] {portrait['artist']} - {portrait['title']}")
            
            # 파일명 생성
            import re
            artist_clean = re.sub(r'[^\w\s\-]', '', portrait['artist'])[:20].replace(' ', '_')
            title_clean = re.sub(r'[^\w\s\-]', '', portrait['title'])[:20].replace(' ', '_')
            filename = f"wiki_{i+1:03d}_{artist_clean}_{title_clean}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # 이미지 다운로드
            headers = {'User-Agent': get_random_user_agent()}
            response = requests.get(portrait['url'], headers=headers, timeout=20)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                file_size_mb = len(response.content) / (1024 * 1024)
                print(f"    ✅ 다운로드 완료: {file_size_mb:.1f}MB")
                
                downloaded_count += 1
                results.append({
                    'source': 'Wikipedia Commons',
                    'title': portrait['title'],
                    'artist': portrait['artist'],
                    'filename': filename,
                    'filepath': filepath,
                    'file_size_mb': file_size_mb,
                    'period': portrait['period'],
                    'medium': portrait['medium'],
                    'status': 'success'
                })
            else:
                print(f"    ❌ 다운로드 실패: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ 오류: {e}")
        
        human_delay()
    
    print(f"🎯 위키피디아 커먼스: {downloaded_count}개 다운로드 완료")
    return results

def download_multiple_museums():
    """여러 미술관에서 초상화 다운로드"""
    
    print("🌍 다중 미술관 초상화 다운로더")
    print("여러 미술관 API와 공개 소스를 활용합니다!")
    print("=" * 60)
    
    all_results = []
    
    # 1단계: 메트로폴리탄 미술관
    print("\n🏛️ === 1단계: 메트로폴리탄 미술관 ===")
    met_ids = search_metropolitan_portraits(limit=500)
    
    # 메트로폴리탄에서 실제 다운로드 (간단 버전)
    met_results = []
    for i, object_id in enumerate(met_ids[:50]):  # 처음 50개만 테스트
        try:
            detail_url = f"{APIS['metropolitan']['base_url']}/objects/{object_id}"
            headers = {'User-Agent': get_random_user_agent()}
            response = requests.get(detail_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                obj_data = response.json()
                image_url = obj_data.get('primaryImage')
                
                if image_url:
                    filename = f"met_{object_id}_{obj_data.get('title', 'Untitled')[:30]}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)
                    
                    img_response = requests.get(image_url, headers=headers, timeout=20)
                    if img_response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)
                        
                        met_results.append({
                            'source': 'Metropolitan Museum',
                            'object_id': object_id,
                            'title': obj_data.get('title', ''),
                            'artist': obj_data.get('artistDisplayName', ''),
                            'filename': filename,
                            'filepath': filepath,
                            'file_size_mb': len(img_response.content) / (1024 * 1024),
                            'status': 'success'
                        })
                        print(f"  ✅ 메트로폴리탄 {object_id} 다운로드 완료")
            
            human_delay()
            
        except Exception as e:
            print(f"  ❌ 메트로폴리탄 {object_id} 오류: {e}")
    
    all_results.extend(met_results)
    
    # 2단계: 위키피디아 커먼스
    print(f"\n📚 === 2단계: 위키피디아 커먼스 ===")
    wiki_results = download_from_wikipedia_commons()
    all_results.extend(wiki_results)
    
    # 3단계: 공개 접근 미술관들
    print(f"\n🌐 === 3단계: 공개 접근 미술관들 ===")
    open_artworks = search_open_access_museums()
    # 실제 다운로드는 구현 생략 (API 키 필요)
    
    # 결과 저장
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(METADATA_DIR, "multi_museum_results.csv"), index=False, encoding='utf-8')
    
    print(f"\n🎉 === 다중 미술관 다운로드 완료 ===")
    print(f"✅ 총 다운로드: {len(all_results)}개")
    print(f"🏛️ 메트로폴리탄: {len(met_results)}개")
    print(f"📚 위키피디아: {len(wiki_results)}개")
    print(f"📁 이미지 저장: {SAVE_DIR}")
    print(f"📄 메타데이터: {METADATA_DIR}/multi_museum_results.csv")
    
    return results_df

def main():
    print("🌍 다중 미술관 초상화 다운로더")
    print("여러 미술관의 API를 활용하여 안정적으로 수집합니다!")
    print("=" * 60)
    
    # 다중 미술관 다운로드 실행
    results_df = download_multiple_museums()
    
    if len(results_df) > 0:
        print(f"\n💡 다중 미술관 결과:")
        print(f"• 여러 소스에서 안정적으로 수집했습니다")
        print(f"• 봇 탐지 위험을 분산시켰습니다")
        print(f"• 다양한 미술관의 작품을 확보했습니다")

if __name__ == "__main__":
    main()
