import os
import requests
import pandas as pd
import time
import re
from urllib.parse import urljoin

# 설정
CSV_FILE = "portraits_dataset.csv"
SAVE_DIR = "data/portrait_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_portraits_from_public_sources():
    """공개 소스에서 초상화 이미지들을 다운로드합니다."""
    
    # 검증된 공개 이미지 URL들 (Wikipedia Commons, 공개 도메인)
    portrait_urls = [
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
            'title': 'Self-Portrait with Bandaged Ear',
            'artist': 'Vincent van Gogh',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/9/95/Vincent_van_Gogh_-_Self-Portrait_with_Bandaged_Ear_-_Google_Art_Project.jpg',
            'period': '1889',
            'medium': 'Oil on canvas'
        },
        {
            'title': 'Portrait of Adele Bloch-Bauer I',
            'artist': 'Gustav Klimt',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/4/40/The_Kiss_-_Gustav_Klimt_-_Google_Art_Project.jpg',
            'period': '1907-1908',
            'medium': 'Oil, silver and gold leaf on canvas'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Frida Kahlo',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/06/Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg',
            'period': '1932',
            'medium': 'Oil on canvas'
        },
        {
            'title': 'Portrait of Madame X',
            'artist': 'John Singer Sargent',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/3/3f/John_Singer_Sargent_-_Madame_X_%28Madame_Pierre_Gautreau%29_-_Google_Art_Project.jpg',
            'period': '1884',
            'medium': 'Oil on canvas'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Rembrandt',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Self-portrait_by_Rembrandt.jpg',
            'period': '1660',
            'medium': 'Oil on canvas'
        },
        {
            'title': 'Portrait of a Man',
            'artist': 'Jan van Eyck',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/7/76/Jan_van_Eyck_-_Portrait_of_a_Man_%28Self_Portrait%29_-_WGA07761.jpg',
            'Thomas': '1433',
            'medium': 'Oil on wood'
        },
        {
            'title': 'The Birth of Venus',
            'artist': 'Sandro Botticelli',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/0b/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg',
            'period': '1485-1486',
            'medium': 'Tempera on canvas'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Pablo Picasso',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8c/Pablo_Picasso%2C_1901%2C_Autoportrait_%28Self-portrait%29%2C_oil_on_cardboard_mounted_on_canvas%2C_73_x_60_cm%2C_Mus%C3%A9e_national_Picasso-Paris.jpg',
            'period': '1901',
            'medium': 'Oil on cardboard'
        }
    ]
    
    downloaded_count = 0
    failed_count = 0
    results = []
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; PortraitDownloader/1.0)",
        "Accept": "image/*"
    })
    
    print("=== 유명 초상화 이미지 다운로드 시작 ===")
    
    for i, portrait in enumerate(portrait_urls):
        print(f"\n[{i+1}/{len(portrait_urls)}] 처리 중: {portrait['title']} - {portrait['artist']}")
        
        # 파일명 생성 (특수문자 제거)
        artist_clean = re.sub(r"[^\w\s\-]", "", portrait['artist']).replace(" ", "_")[:30]
        title_clean = re.sub(r"[^\w\s\-]", "", portrait['title']).replace(" ", "_")[:30]
        filename = f"{i+1:03d}_{artist_clean}_{title_clean}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # 이미지가 이미 존재하는지 확인
        if os.path.exists(filepath):
            print(f"✓ 이미 존재: {filename}")
            results.append({
                'filename': filename,
                'title': portrait['title'],
                'artist': portrait['artist'],
                'period': portrait.get('period', ''),
                'medium': portrait.get('medium', ''),
                'url': portrait['url'],
                'filepath': filepath,
                'status': 'already_exists'
            })
            downloaded_count += 1
            continue
        
        try:
            # 이미지 다운로드
            response = session.get(portrait['url'], timeout=30)
            
            if response.status_code == 200 and response.content:
                # 파일 크기 확인 (최소 10KB)
                if len(response.content) > 10240:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✓ 다운로드 완료: {filename} ({len(response.content)/1024:.1f}KB)")
                    downloaded_count += 1
                    results.append({
                        'filename': filename,
                        'title': portrait['title'],
                        'artist': portrait['artist'],
                        'period': portrait.get('period', ''),
                        'medium': portrait.get('medium', ''),
                        'url': portrait['url'],
                        'filepath': filepath,
                        'status': 'success',
                        'file_size_kb': len(response.content)/1024
                    })
                else:
                    print(f"✗ 파일 크기가 너무 작음: {filename}")
                    failed_count += 1
                    results.append({
                        'filename': filename,
                        'title': portrait['title'],
                        'artist': portrait['artist'],
                        'url': portrait['url'],
                        'filepath': '',
                        'status': 'file_too_small'
                    })
            else:
                print(f"✗ HTTP 오류 {response.status_code}: {filename}")
                failed_count += 1
                results.append({
                    'filename': filename,
                    'title': portrait['title'],
                    'artist': portrait['artist'],
                    'url': portrait['url'],
                    'filepath': '',
                    'status': f'http_error_{response.status_code}'
                })
                
        except Exception as e:
            print(f"✗ 오류 발생: {filename} - {str(e)}")
            failed_count += 1
            results.append({
                'filename': filename,
                'title': portrait['title'],
                'artist': portrait['artist'],
                'url': portrait['url'],
                'filepath': '',
                'status': f'error_{type(e).__name__}'
            })
        
        # API 호출 제한을 위한 대기
        time.sleep(1)
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("downloaded_portraits_results.csv", index=False, encoding='utf-8')
    
    print(f"\n=== 다운로드 완료 ===")
    print(f"성공: {downloaded_count}개")
    print(f"실패: {failed_count}개")
    print(f"총 처리: {len(portrait_urls)}개")
    print(f"결과 파일: downloaded_portraits_results.csv")
    print(f"이미지 저장 위치: {SAVE_DIR}")
    
    return results_df

def create_training_dataset():
    """다운로드된 이미지들로 훈련용 데이터셋을 생성합니다."""
    
    results_file = "downloaded_portraits_results.csv"
    if not os.path.exists(results_file):
        print("결과 파일이 없습니다. 먼저 이미지를 다운로드하세요.")
        return
    
    df = pd.read_csv(results_file)
    successful_downloads = df[df['status'].isin(['success', 'already_exists'])]
    
    if len(successful_downloads) == 0:
        print("다운로드된 이미지가 없습니다.")
        return
    
    # 훈련용 데이터셋 생성
    training_data = []
    
    for _, row in successful_downloads.iterrows():
        if os.path.exists(row['filepath']):
            training_data.append({
                'image_path': row['filepath'],
                'title': row['title'],
                'artist': row['artist'],
                'period': row.get('period', ''),
                'medium': row.get('medium', ''),
                'style': f"{row['artist']} style",  # 미술가별 스타일 라벨
                'genre': 'Portrait',
                'file_size_kb': row.get('file_size_kb', 0)
            })
    
    training_df = pd.DataFrame(training_data)
    training_df.to_csv("portrait_training_dataset.csv", index=False, encoding='utf-8')
    
    print(f"\n=== 훈련용 데이터셋 생성 완료 ===")
    print(f"총 이미지: {len(training_df)}개")
    print(f"데이터셋 파일: portrait_training_dataset.csv")
    print(f"포함된 미술가: {', '.join(training_df['artist'].unique())}")

def main():
    print("🎨 초상화 이미지 다운로더 및 훈련용 데이터셋 생성기")
    print("=" * 60)
    
    # 1. 이미지 다운로드
    results_df = download_portraits_from_public_sources()
    
    # 2. 훈련용 데이터셋 생성
    if len(results_df) > 0:
        create_training_dataset()
    
    print("\n💡 다음 단계:")
    print("1. portrait_training_dataset.csv 파일을 확인하세요")
    print("2. 각 미술가의 스타일 특징을 분석하세요")
    print("3. AI 모델 훈련을 위한 추가 데이터를 준비하세요")
    print("4. 관객의 GPT 데이터와 결합하여 자화상 생성 모델을 개발하세요")

if __name__ == "__main__":
    main()
