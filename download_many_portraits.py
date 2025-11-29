import os
import requests
import pandas as pd
import time
import re
from urllib.parse import urljoin

# 설정
SAVE_DIR = "data/many_portraits"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_many_portraits_from_wikipedia():
    """Wikipedia Commons에서 많은 초상화들을 다운로드합니다."""
    
    # Wikipedia Commons의 유명한 초상화들 (검증된 URL들)
    portrait_urls = [
        # 레오나르도 다 빈치
        {
            'title': 'Mona Lisa',
            'artist': 'Leonardo da Vinci',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg',
            'period': '1503-1519',
            'medium': 'Oil on poplar panel',
            'style': 'Renaissance'
        },
        {
            'title': 'Portrait of a Man in Red Chalk',
            'artist': 'Leonardo da Vinci',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/4/4b/Leonardo_self.jpg',
            'period': '1512',
            'medium': 'Red chalk on paper',
            'style': 'Renaissance'
        },
        
        # 요하네스 베르메르
        {
            'title': 'Girl with a Pearl Earring',
            'artist': 'Johannes Vermeer',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/0f/1665_Girl_with_a_Pearl_Earring.jpg',
            'period': '1665',
            'medium': 'Oil on canvas',
            'style': 'Dutch Golden Age'
        },
        
        # 반 고흐
        {
            'title': 'Self-Portrait',
            'artist': 'Vincent van Gogh',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/b/b2/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg',
            'period': '1889',
            'medium': 'Oil on canvas',
            'style': 'Post-Impressionism'
        },
        {
            'title': 'Portrait of Dr. Gachet',
            'artist': 'Vincent van Gogh',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/4/4a/Vincent_van_Gogh_-_Portrait_of_Dr._Gachet_%281st_version%29_-_Google_Art_Project.jpg',
            'period': '1890',
            'medium': 'Oil on canvas',
            'style': 'Post-Impressionism'
        },
        
        # 프리다 칼로
        {
            'title': 'Self-Portrait',
            'artist': 'Frida Kahlo',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/06/Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg',
            'period': '1932',
            'medium': 'Oil on canvas',
            'style': 'Surrealism'
        },
        {
            'title': 'Self-Portrait with Thorn Necklace and Hummingbird',
            'artist': 'Frida Kahlo',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/4/4e/Frida_Kahlo%2C_by_Guillermo_Kahlo_2.jpg',
            'period': '1940',
            'medium': 'Oil on canvas',
            'style': 'Surrealism'
        },
        
        # 렘브란트
        {
            'title': 'Self-Portrait',
            'artist': 'Rembrandt',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Self-portrait_by_Rembrandt.jpg',
            'period': '1660',
            'medium': 'Oil on canvas',
            'style': 'Dutch Golden Age'
        },
        {
            'title': 'The Night Watch',
            'artist': 'Rembrandt',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/5/5a/The_Night_Watch_-_HD.jpg',
            'period': '1642',
            'medium': 'Oil on canvas',
            'style': 'Dutch Golden Age'
        },
        
        # 산드로 보티첼리
        {
            'title': 'The Birth of Venus',
            'artist': 'Sandro Botticelli',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/0b/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg',
            'period': '1485-1486',
            'medium': 'Tempera on canvas',
            'style': 'Early Renaissance'
        },
        
        # 얀 반 에이크
        {
            'title': 'Portrait of a Man',
            'artist': 'Jan van Eyck',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/7/76/Jan_van_Eyck_-_Portrait_of_a_Man_%28Self_Portrait%29_-_WGA07761.jpg',
            'period': '1433',
            'medium': 'Oil on wood',
            'style': 'Northern Renaissance'
        },
        
        # 피카소
        {
            'title': 'Self-Portrait',
            'artist': 'Pablo Picasso',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8c/Pablo_Picasso%2C_1901%2C_Autoportrait_%28Self-portrait%29%2C_oil_on_cardboard_mounted_on_canvas%2C_73_x_60_cm%2C_Mus%C3%A9e_national_Picasso-Paris.jpg',
            'period': '1901',
            'medium': 'Oil on cardboard',
            'style': 'Blue Period'
        },
        {
            'title': 'Les Demoiselles d\'Avignon',
            'artist': 'Pablo Picasso',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/Les_Demoiselles_d%27Avignon.jpg',
            'period': '1907',
            'medium': 'Oil on canvas',
            'style': 'Cubism'
        },
        
        # 클림트
        {
            'title': 'The Kiss',
            'artist': 'Gustav Klimt',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/4/40/The_Kiss_-_Gustav_Klimt_-_Google_Art_Project.jpg',
            'period': '1907-1908',
            'medium': 'Oil, silver and gold leaf on canvas',
            'style': 'Art Nouveau'
        },
        
        # 모네
        {
            'title': 'Self-Portrait',
            'artist': 'Claude Monet',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8c/Claude_Monet_1889_Photo_by_Nadar.jpg',
            'period': '1889',
            'medium': 'Photograph',
            'style': 'Impressionism'
        },
        
        # 마네
        {
            'title': 'Olympia',
            'artist': 'Édouard Manet',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/0f/1863_Manet_Olympia.jpg',
            'period': '1863',
            'medium': 'Oil on canvas',
            'style': 'Realism'
        },
        
        # 고갱
        {
            'title': 'Self-Portrait',
            'artist': 'Paul Gauguin',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/6/63/Paul_Gauguin_1891%2C_Self-portrait%2C_oil_on_canvas%2C_92_x_73_cm%2C_Mus%C3%A9e_d%27Orsay%2C_Paris.jpg',
            'period': '1891',
            'medium': 'Oil on canvas',
            'style': 'Post-Impressionism'
        },
        
        # 세잔느
        {
            'title': 'Self-Portrait',
            'artist': 'Paul Cézanne',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Paul_C%C3%A9zanne_1864_Self-portrait.jpg',
            'period': '1864',
            'medium': 'Oil on canvas',
            'style': 'Post-Impressionism'
        },
        
        # 마티스
        {
            'title': 'Self-Portrait',
            'artist': 'Henri Matisse',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Henri_Matisse%2C_1906%2C_Self-Portrait_in_a_Striped_T-shirt%2C_oil_on_canvas%2C_55_x_46_cm%2C_Statens_Museum_for_Kunst%2C_Copenhagen.jpg',
            'period': '1906',
            'medium': 'Oil on canvas',
            'style': 'Fauvism'
        },
        
        # 달리
        {
            'title': 'Self-Portrait',
            'artist': 'Salvador Dalí',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Salvador_Dal%C3%AD_1939_Self-portrait.jpg',
            'period': '1939',
            'medium': 'Oil on canvas',
            'style': 'Surrealism'
        },
        
        # 앤디 워홀
        {
            'title': 'Self-Portrait',
            'artist': 'Andy Warhol',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/2/2e/Andy_Warhol_by_Jack_Mitchell.jpg',
            'period': '1970',
            'medium': 'Photograph',
            'style': 'Pop Art'
        },
        
        # 바스키아
        {
            'title': 'Self-Portrait',
            'artist': 'Jean-Michel Basquiat',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Jean-Michel_Basquiat_1985.jpg',
            'period': '1985',
            'medium': 'Photograph',
            'style': 'Neo-Expressionism'
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
    
    print("🎨 Wikipedia Commons에서 대량 초상화 다운로드 시작!")
    print(f"총 {len(portrait_urls)}개의 작품을 처리합니다.")
    print("=" * 60)
    
    for i, portrait in enumerate(portrait_urls):
        print(f"\n[{i+1}/{len(portrait_urls)}] 처리 중: {portrait['title']} - {portrait['artist']}")
        
        # 파일명 생성 (특수문자 제거)
        artist_clean = re.sub(r"[^\w\s\-]", "", portrait['artist']).replace(" ", "_")[:25]
        title_clean = re.sub(r"[^\w\s\-]", "", portrait['title']).replace(" ", "_")[:25]
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
                'style': portrait.get('style', ''),
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
                    
                    file_size_mb = len(response.content) / (1024 * 1024)
                    print(f"✓ 다운로드 완료: {filename} ({file_size_mb:.1f}MB)")
                    downloaded_count += 1
                    results.append({
                        'filename': filename,
                        'title': portrait['title'],
                        'artist': portrait['artist'],
                        'period': portrait.get('period', ''),
                        'medium': portrait.get('medium', ''),
                        'style': portrait.get('style', ''),
                        'url': portrait['url'],
                        'filepath': filepath,
                        'status': 'success',
                        'file_size_mb': file_size_mb
                    })
                else:
                    print(f"✗ 파일 크기가 너무 작음: {filename}")
                    failed_count += 1
            else:
                print(f"✗ HTTP 오류 {response.status_code}: {filename}")
                failed_count += 1
                
        except Exception as e:
            print(f"✗ 오류 발생: {filename} - {str(e)}")
            failed_count += 1
        
        # API 호출 제한을 위한 대기
        time.sleep(1)
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("many_portraits_download_results.csv", index=False, encoding='utf-8')
    
    print(f"\n🎉 === 다운로드 완료 ===")
    print(f"✅ 성공: {downloaded_count}개")
    print(f"❌ 실패: {failed_count}개")
    print(f"📊 총 처리: {len(portrait_urls)}개")
    print(f"📁 결과 파일: many_portraits_download_results.csv")
    print(f"🖼️ 이미지 저장 위치: {SAVE_DIR}")
    
    return results_df

def analyze_downloaded_styles(results_df):
    """다운로드된 작품들의 스타일을 분석합니다."""
    
    if len(results_df) == 0:
        print("분석할 데이터가 없습니다.")
        return
    
    print(f"\n🎨 === 스타일별 분석 ===")
    
    # 스타일별 작품 수
    style_counts = results_df['style'].value_counts()
    print(f"\n📊 스타일별 작품 수:")
    for style, count in style_counts.items():
        print(f"  • {style}: {count}개")
    
    # 미술가별 작품 수
    artist_counts = results_df['artist'].value_counts()
    print(f"\n👨‍🎨 미술가별 작품 수:")
    for artist, count in artist_counts.items():
        print(f"  • {artist}: {count}개")
    
    # 시대별 작품 수
    period_counts = results_df['period'].value_counts()
    print(f"\n⏰ 시대별 작품 수:")
    for period, count in period_counts.items():
        print(f"  • {period}: {count}개")

def main():
    print("🚀 대량 초상화 다운로더")
    print("Wikipedia Commons에서 유명한 초상화들을 다운로드합니다!")
    print("=" * 60)
    
    # 다운로드 실행
    results_df = download_many_portraits_from_wikipedia()
    
    # 분석
    if len(results_df) > 0:
        analyze_downloaded_styles(results_df)
        
        # 훈련용 데이터셋 생성
        training_data = []
        for _, row in results_df.iterrows():
            if os.path.exists(row['filepath']):
                training_data.append({
                    'image_path': row['filepath'],
                    'title': row['title'],
                    'artist': row['artist'],
                    'period': row['period'],
                    'medium': row['medium'],
                    'style': row['style'],
                    'genre': 'Portrait',
                    'file_size_mb': row.get('file_size_mb', 0)
                })
        
        training_df = pd.DataFrame(training_data)
        training_df.to_csv("large_portrait_training_dataset.csv", index=False, encoding='utf-8')
        
        print(f"\n📚 훈련용 데이터셋 생성 완료!")
        print(f"📄 파일: large_portrait_training_dataset.csv")
        print(f"🖼️ 총 이미지: {len(training_df)}개")
        
        print(f"\n💡 다음 단계:")
        print(f"1. 각 미술가의 스타일 특징을 분석하세요")
        print(f"2. AI 모델 훈련을 위한 전처리를 진행하세요")
        print(f"3. 관객의 GPT 데이터와 결합하여 자화상 생성 모델을 개발하세요")

if __name__ == "__main__":
    main()
