import os
import requests
import pandas as pd
import time
from urllib.parse import urljoin

# 설정
SAVE_DIR = "data/public_portraits"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_wikiart_portraits():
    """WikiArt에서 초상화 다운로드 (공개 데이터셋)"""
    
    # WikiArt 초상화 카테고리 URL들
    portrait_urls = [
        "https://uploads6.wikiart.org/images/leonardo-da-vinci/mona-lisa-1503.jpg",
        "https://uploads2.wikiart.org/images/vincent-van-gogh/self-portrait-1889-1.jpg",
        "https://uploads3.wikiart.org/images/jan-van-eyck/portrait-of-a-man-1433.jpg",
        "https://uploads7.wikiart.org/images/rembrandt/self-portrait-1660.jpg",
        "https://uploads1.wikiart.org/images/johannes-vermeer/girl-with-a-pearl-earring-1665.jpg",
        "https://uploads8.wikiart.org/images/pablo-picasso/self-portrait-1907.jpg",
        "https://uploads4.wikiart.org/images/frida-kahlo/self-portrait-with-thorn-necklace-and-hummingbird-1940.jpg",
        "https://uploads5.wikiart.org/images/andy-warhol/self-portrait-1966.jpg",
        "https://uploads9.wikiart.org/images/sandro-botticelli/portrait-of-a-young-man-1480.jpg",
        "https://uploads0.wikiart.org/images/raphael/self-portrait-1506.jpg"
    ]
    
    downloaded_count = 0
    failed_count = 0
    results = []
    
    for i, url in enumerate(portrait_urls):
        try:
            # 파일명 생성
            filename = f"portrait_{i+1:03d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # 이미지 다운로드
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; PortraitDownloader/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=20)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ 다운로드 완료: {filename}")
                downloaded_count += 1
                results.append({
                    'filename': filename,
                    'url': url,
                    'filepath': filepath,
                    'status': 'success'
                })
            else:
                print(f"✗ 다운로드 실패: {filename} (상태코드: {response.status_code})")
                failed_count += 1
                results.append({
                    'filename': filename,
                    'url': url,
                    'filepath': '',
                    'status': 'failed'
                })
                
        except Exception as e:
            print(f"✗ 오류 발생: {filename} - {e}")
            failed_count += 1
            results.append({
                'filename': filename,
                'url': url,
                'filepath': '',
                'status': 'error'
            })
        
        # 요청 간격 조절
        time.sleep(1)
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("wikiart_download_results.csv", index=False, encoding='utf-8')
    
    print(f"\n=== 다운로드 완료 ===")
    print(f"성공: {downloaded_count}개")
    print(f"실패: {failed_count}개")
    print(f"결과 파일: wikiart_download_results.csv")
    print(f"이미지 저장 위치: {SAVE_DIR}")

def download_from_met_public_collection():
    """메트로폴리탄 미술관 공개 컬렉션에서 초상화 다운로드"""
    
    # 메트로폴리탄 미술관 공개 이미지 URL들 (초상화)
    met_portraits = [
        {
            'title': 'Portrait of a Man',
            'artist': 'Unknown Artist',
            'url': 'https://images.metmuseum.org/CRDImages/ep/original/DP145909.jpg'
        },
        {
            'title': 'Portrait of a Woman',
            'artist': 'Unknown Artist', 
            'url': 'https://images.metmuseum.org/CRDImages/ep/original/DP145910.jpg'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Rembrandt',
            'url': 'https://images.metmuseum.org/CRDImages/ep/original/DP145911.jpg'
        }
    ]
    
    downloaded_count = 0
    failed_count = 0
    results = []
    
    for i, portrait in enumerate(met_portraits):
        try:
            # 파일명 생성
            artist_name = portrait['artist'].replace(' ', '_')[:20]
            title_name = portrait['title'].replace(' ', '_')[:20]
            filename = f"met_{i+1:03d}_{artist_name}_{title_name}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # 이미지 다운로드
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; PortraitDownloader/1.0)'
            }
            response = requests.get(portrait['url'], headers=headers, timeout=20)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ 다운로드 완료: {filename}")
                downloaded_count += 1
                results.append({
                    'filename': filename,
                    'title': portrait['title'],
                    'artist': portrait['artist'],
                    'url': portrait['url'],
                    'filepath': filepath,
                    'status': 'success'
                })
            else:
                print(f"✗ 다운로드 실패: {filename} (상태코드: {response.status_code})")
                failed_count += 1
                
        except Exception as e:
            print(f"✗ 오류 발생: {filename} - {e}")
            failed_count += 1
        
        time.sleep(1)
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("met_download_results.csv", index=False, encoding='utf-8')
    
    print(f"\n=== 메트로폴리탄 다운로드 완료 ===")
    print(f"성공: {downloaded_count}개")
    print(f"실패: {failed_count}개")
    print(f"결과 파일: met_download_results.csv")

def main():
    print("=== 공개 미술 데이터셋에서 초상화 다운로드 ===")
    print("1. WikiArt에서 유명 초상화 다운로드")
    download_wikiart_portraits()
    
    print("\n2. 메트로폴리탄 미술관 공개 컬렉션에서 다운로드")
    download_from_met_public_collection()

if __name__ == "__main__":
    main()
