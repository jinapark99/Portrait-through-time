#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대안 미술관에서 초상화 다운로드
- Rijksmuseum (네덜란드)
- Europeana
- 기타 공개 미술관 API
"""

import requests
import os
import time
import random
import re
import json
from urllib.parse import urljoin, urlparse
import pandas as pd
from datetime import datetime

# 설정
SAVE_DIR = "data/alternative_portraits"
METADATA_DIR = "data/alternative_metadata"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# API 엔드포인트들
RIJKSMUSEUM_API = "https://www.rijksmuseum.nl/api/nl/collection"
EUROPEANA_API = "https://www.europeana.eu/api/v2/search.json"

# User-Agent 목록
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def get_random_headers():
    """랜덤 헤더 생성"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

def human_delay():
    """인간적인 지연"""
    delay = random.uniform(1.0, 2.0)
    time.sleep(delay)

def search_rijksmuseum_portraits(limit=100):
    """Rijksmuseum에서 초상화 검색"""
    print(f"🔍 Rijksmuseum에서 초상화 검색 중... (목표: {limit}개)")
    
    all_objects = []
    page = 1
    per_page = 100
    
    session = requests.Session()
    
    while len(all_objects) < limit:
        try:
            params = {
                "key": "YOUR_API_KEY",  # Rijksmuseum API 키 필요
                "format": "json",
                "type": "schilderij",  # 그림
                "q": "portret",  # 초상화
                "ps": min(per_page, limit - len(all_objects)),
                "p": page
            }
            
            print(f"  📡 페이지 {page} 요청 중...")
            
            response = session.get(
                RIJKSMUSEUM_API,
                params=params,
                headers=get_random_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                objects = data.get("artObjects", [])
                
                if not objects:
                    print("  ✅ 더 이상 결과가 없습니다.")
                    break
                
                all_objects.extend(objects)
                print(f"  ✅ {len(objects)}개 발견 (총 {len(all_objects)}개)")
                
                page += 1
                human_delay()
                
            else:
                print(f"  ❌ API 오류: {response.status_code}")
                break
                
        except Exception as e:
            print(f"  ❌ 검색 오류: {e}")
            break
    
    print(f"🎉 Rijksmuseum에서 {len(all_objects)}개의 초상화 작품 발견!")
    return all_objects[:limit]

def search_europeana_portraits(limit=100):
    """Europeana에서 초상화 검색"""
    print(f"🔍 Europeana에서 초상화 검색 중... (목표: {limit}개)")
    
    all_objects = []
    start = 0
    rows = 100
    
    session = requests.Session()
    
    while len(all_objects) < limit:
        try:
            params = {
                "wskey": "YOUR_API_KEY",  # Europeana API 키 필요
                "query": "portrait",
                "qf": "TYPE:IMAGE",
                "start": start,
                "rows": min(rows, limit - len(all_objects)),
                "profile": "standard"
            }
            
            print(f"  📡 페이지 {start//rows + 1} 요청 중...")
            
            response = session.get(
                EUROPEANA_API,
                params=params,
                headers=get_random_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                objects = data.get("items", [])
                
                if not objects:
                    print("  ✅ 더 이상 결과가 없습니다.")
                    break
                
                all_objects.extend(objects)
                print(f"  ✅ {len(objects)}개 발견 (총 {len(all_objects)}개)")
                
                start += rows
                human_delay()
                
            else:
                print(f"  ❌ API 오류: {response.status_code}")
                break
                
        except Exception as e:
            print(f"  ❌ 검색 오류: {e}")
            break
    
    print(f"🎉 Europeana에서 {len(all_objects)}개의 초상화 작품 발견!")
    return all_objects[:limit]

def download_sample_portraits():
    """샘플 초상화 다운로드 (API 키 없이)"""
    print("🎨 샘플 초상화 다운로드 (공개 도메인 이미지)")
    
    # 공개 도메인 초상화 URL들 (대량 수집용)
    portrait_urls = [
        # 레오나르도 다 빈치
        {
            "title": "Mona Lisa",
            "artist": "Leonardo da Vinci",
            "url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
            "year": "1503-1519"
        },
        {
            "title": "Self-Portrait",
            "artist": "Leonardo da Vinci",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Leonardo_self.jpg/687px-Leonardo_self.jpg",
            "year": "1512"
        },
        
        # 요하네스 베르메르
        {
            "title": "Girl with a Pearl Earring",
            "artist": "Johannes Vermeer",
            "url": "https://upload.wikimedia.org/wikipedia/commons/0/0f/1665_Girl_with_a_Pearl_Earring.jpg",
            "year": "1665"
        },
        {
            "title": "The Milkmaid",
            "artist": "Johannes Vermeer",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Vermeer_-_The_Milkmaid_-_Google_Art_Project.jpg/687px-Vermeer_-_The_Milkmaid_-_Google_Art_Project.jpg",
            "year": "1658"
        },
        
        # 빈센트 반 고흐
        {
            "title": "Self-Portrait",
            "artist": "Vincent van Gogh",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg/687px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg",
            "year": "1889"
        },
        {
            "title": "Self-Portrait with Bandaged Ear",
            "artist": "Vincent van Gogh",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Vincent_van_Gogh_-_Self-Portrait_with_Bandaged_Ear_-_Google_Art_Project.jpg/687px-Vincent_van_Gogh_-_Self-Portrait_with_Bandaged_Ear_-_Google_Art_Project.jpg",
            "year": "1889"
        },
        {
            "title": "Portrait of Dr. Gachet",
            "artist": "Vincent van Gogh",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Vincent_van_Gogh_-_Portrait_of_Dr._Gachet_-_Google_Art_Project.jpg/687px-Vincent_van_Gogh_-_Portrait_of_Dr._Gachet_-_Google_Art_Project.jpg",
            "year": "1890"
        },
        
        # 렘브란트
        {
            "title": "Self-Portrait",
            "artist": "Rembrandt",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Self-portrait_by_Rembrandt.jpg/687px-Self-portrait_by_Rembrandt.jpg",
            "year": "1659"
        },
        {
            "title": "The Night Watch",
            "artist": "Rembrandt",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/The_Night_Watch_-_Rembrandt_van_Rijn_-_Google_Art_Project.jpg/687px-The_Night_Watch_-_Rembrandt_van_Rijn_-_Google_Art_Project.jpg",
            "year": "1642"
        },
        {
            "title": "Portrait of Jan Six",
            "artist": "Rembrandt",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Rembrandt_-_Portrait_of_Jan_Six_-_Google_Art_Project.jpg/687px-Rembrandt_-_Portrait_of_Jan_Six_-_Google_Art_Project.jpg",
            "year": "1654"
        },
        
        # 얀 반 에이크
        {
            "title": "Portrait of a Man",
            "artist": "Jan van Eyck",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Jan_van_Eyck_1433_Man_in_a_Red_Turban.jpg/687px-Jan_van_Eyck_1433_Man_in_a_Red_Turban.jpg",
            "year": "1433"
        },
        {
            "title": "The Arnolfini Portrait",
            "artist": "Jan van Eyck",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Jan_van_Eyck_-_The_Arnolfini_Portrait_-_Google_Art_Project.jpg/687px-Jan_van_Eyck_-_The_Arnolfini_Portrait_-_Google_Art_Project.jpg",
            "year": "1434"
        },
        
        # 프리다 칼로
        {
            "title": "Self-Portrait",
            "artist": "Frida Kahlo",
            "url": "https://upload.wikimedia.org/wikipedia/commons/0/06/Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg",
            "year": "1932"
        },
        {
            "title": "Self-Portrait with Thorn Necklace",
            "artist": "Frida Kahlo",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Frida_Kahlo_-_Self-Portrait_with_Thorn_Necklace_and_Hummingbird_-_Google_Art_Project.jpg/687px-Frida_Kahlo_-_Self-Portrait_with_Thorn_Necklace_and_Hummingbird_-_Google_Art_Project.jpg",
            "year": "1940"
        },
        
        # 알브레히트 뒤러
        {
            "title": "Self-Portrait",
            "artist": "Albrecht Dürer",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Albrecht_D%C3%BCrer_-_Self-Portrait_-_WGA06755.jpg/687px-Albrecht_D%C3%BCrer_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1500"
        },
        {
            "title": "Portrait of a Young Man",
            "artist": "Albrecht Dürer",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Albrecht_D%C3%BCrer_-_Portrait_of_a_Young_Man_-_WGA06755.jpg/687px-Albrecht_D%C3%BCrer_-_Portrait_of_a_Young_Man_-_WGA06755.jpg",
            "year": "1500"
        },
        
        # 산드로 보티첼리
        {
            "title": "Portrait of a Young Man",
            "artist": "Sandro Botticelli",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Sandro_Botticelli_-_Portrait_of_a_Young_Man_-_WGA02812.jpg/687px-Sandro_Botticelli_-_Portrait_of_a_Young_Man_-_WGA02812.jpg",
            "year": "1480-1485"
        },
        {
            "title": "Portrait of a Lady",
            "artist": "Sandro Botticelli",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Sandro_Botticelli_-_Portrait_of_a_Lady_-_WGA06755.jpg/687px-Sandro_Botticelli_-_Portrait_of_a_Lady_-_WGA06755.jpg",
            "year": "1480"
        },
        
        # 한스 홀바인
        {
            "title": "Portrait of a Man",
            "artist": "Hans Holbein the Younger",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Hans_Holbein_the_Younger_-_Portrait_of_a_Man_-_WGA11577.jpg/687px-Hans_Holbein_the_Younger_-_Portrait_of_a_Man_-_WGA11577.jpg",
            "year": "1530"
        },
        {
            "title": "The Ambassadors",
            "artist": "Hans Holbein the Younger",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Hans_Holbein_the_Younger_-_The_Ambassadors_-_Google_Art_Project.jpg/687px-Hans_Holbein_the_Younger_-_The_Ambassadors_-_Google_Art_Project.jpg",
            "year": "1533"
        },
        
        # 파블로 피카소
        {
            "title": "Self-Portrait",
            "artist": "Pablo Picasso",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Pablo_picasso_1.jpg/687px-Pablo_picasso_1.jpg",
            "year": "1907"
        },
        {
            "title": "Portrait of Dora Maar",
            "artist": "Pablo Picasso",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Pablo_Picasso_-_Portrait_of_Dora_Maar_-_Google_Art_Project.jpg/687px-Pablo_Picasso_-_Portrait_of_Dora_Maar_-_Google_Art_Project.jpg",
            "year": "1937"
        },
        
        # 티치아노
        {
            "title": "Portrait of a Woman",
            "artist": "Titian",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Titian_-_Portrait_of_a_Woman_-_WGA22988.jpg/687px-Titian_-_Portrait_of_a_Woman_-_WGA22988.jpg",
            "year": "1515"
        },
        {
            "title": "Portrait of a Man",
            "artist": "Titian",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Titian_-_Portrait_of_a_Man_-_WGA06755.jpg/687px-Titian_-_Portrait_of_a_Man_-_WGA06755.jpg",
            "year": "1515"
        },
        
        # 디에고 벨라스케스
        {
            "title": "Self-Portrait",
            "artist": "Diego Velázquez",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Diego_Vel%C3%A1zquez_-_Self-Portrait_-_WGA24403.jpg/687px-Diego_Vel%C3%A1zquez_-_Self-Portrait_-_WGA24403.jpg",
            "year": "1640"
        },
        {
            "title": "Las Meninas",
            "artist": "Diego Velázquez",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Las_Meninas%2C_by_Diego_Vel%C3%A1zquez%2C_from_Prado_in_Google_Earth.jpg/687px-Las_Meninas%2C_by_Diego_Vel%C3%A1zquez%2C_from_Prado_in_Google_Earth.jpg",
            "year": "1656"
        },
        
        # 라파엘로
        {
            "title": "Portrait of a Man",
            "artist": "Raphael",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Raphael_-_Portrait_of_a_Man_-_WGA18920.jpg/687px-Raphael_-_Portrait_of_a_Man_-_WGA18920.jpg",
            "year": "1515"
        },
        {
            "title": "Self-Portrait",
            "artist": "Raphael",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Raphael_-_Self-Portrait_-_WGA06755.jpg/687px-Raphael_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1506"
        },
        
        # 피터 폴 루벤스
        {
            "title": "Self-Portrait",
            "artist": "Peter Paul Rubens",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Peter_Paul_Rubens_-_Self-Portrait_-_WGA19820.jpg/687px-Peter_Paul_Rubens_-_Self-Portrait_-_WGA19820.jpg",
            "year": "1623"
        },
        {
            "title": "Portrait of Helena Fourment",
            "artist": "Peter Paul Rubens",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Peter_Paul_Rubens_-_Portrait_of_Helena_Fourment_-_WGA06755.jpg/687px-Peter_Paul_Rubens_-_Portrait_of_Helena_Fourment_-_WGA06755.jpg",
            "year": "1630"
        },
        
        # 안토니 반 다이크
        {
            "title": "Portrait of a Lady",
            "artist": "Anthony van Dyck",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Anthony_van_Dyck_-_Portrait_of_a_Lady_-_WGA06755.jpg/687px-Anthony_van_Dyck_-_Portrait_of_a_Lady_-_WGA06755.jpg",
            "year": "1630"
        },
        {
            "title": "Self-Portrait",
            "artist": "Anthony van Dyck",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Anthony_van_Dyck_-_Self-Portrait_-_WGA06755.jpg/687px-Anthony_van_Dyck_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1640"
        },
        
        # 프란스 할스
        {
            "title": "Self-Portrait",
            "artist": "Frans Hals",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Frans_Hals_-_Self-Portrait_-_WGA06755.jpg/687px-Frans_Hals_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1650"
        },
        {
            "title": "Portrait of a Man",
            "artist": "Frans Hals",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Frans_Hals_-_Portrait_of_a_Man_-_WGA06755.jpg/687px-Frans_Hals_-_Portrait_of_a_Man_-_WGA06755.jpg",
            "year": "1630"
        },
        
        # 조슈아 레이놀즈
        {
            "title": "Portrait of a Man",
            "artist": "Joshua Reynolds",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Joshua_Reynolds_-_Portrait_of_a_Man_-_WGA06755.jpg/687px-Joshua_Reynolds_-_Portrait_of_a_Man_-_WGA06755.jpg",
            "year": "1780"
        },
        {
            "title": "Self-Portrait",
            "artist": "Joshua Reynolds",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Joshua_Reynolds_-_Self-Portrait_-_WGA06755.jpg/687px-Joshua_Reynolds_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1780"
        },
        
        # 토마스 게인즈버러
        {
            "title": "Self-Portrait",
            "artist": "Thomas Gainsborough",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Thomas_Gainsborough_-_Self-Portrait_-_WGA06755.jpg/687px-Thomas_Gainsborough_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1787"
        },
        {
            "title": "Portrait of a Lady",
            "artist": "Thomas Gainsborough",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Thomas_Gainsborough_-_Portrait_of_a_Lady_-_WGA06755.jpg/687px-Thomas_Gainsborough_-_Portrait_of_a_Lady_-_WGA06755.jpg",
            "year": "1780"
        },
        
        # 토마스 로렌스
        {
            "title": "Portrait of a Lady",
            "artist": "Thomas Lawrence",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Thomas_Lawrence_-_Portrait_of_a_Lady_-_WGA06755.jpg/687px-Thomas_Lawrence_-_Portrait_of_a_Lady_-_WGA06755.jpg",
            "year": "1820"
        },
        {
            "title": "Self-Portrait",
            "artist": "Thomas Lawrence",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Thomas_Lawrence_-_Self-Portrait_-_WGA06755.jpg/687px-Thomas_Lawrence_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1825"
        },
        
        # 로지에 반 데르 베이덴
        {
            "title": "Portrait of a Lady",
            "artist": "Rogier van der Weyden",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Portrait_of_a_Lady_%28Rogier_van_der_Weyden%29.jpg/687px-Portrait_of_a_Lady_%28Rogier_van_der_Weyden%29.jpg",
            "year": "1460"
        },
        {
            "title": "Portrait of a Man",
            "artist": "Rogier van der Weyden",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Rogier_van_der_Weyden_-_Portrait_of_a_Man_-_WGA06755.jpg/687px-Rogier_van_der_Weyden_-_Portrait_of_a_Man_-_WGA06755.jpg",
            "year": "1460"
        },
        
        # 추가 유명 초상화들
        {
            "title": "Portrait of a Young Woman",
            "artist": "Petrus Christus",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Petrus_Christus_-_Portrait_of_a_Young_Woman_-_WGA06755.jpg/687px-Petrus_Christus_-_Portrait_of_a_Young_Woman_-_WGA06755.jpg",
            "year": "1470"
        },
        {
            "title": "Self-Portrait",
            "artist": "Rembrandt",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Rembrandt_-_Self-Portrait_-_WGA06755.jpg/687px-Rembrandt_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1660"
        },
        {
            "title": "Portrait of a Man",
            "artist": "Hans Memling",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Hans_Memling_-_Portrait_of_a_Man_-_WGA06755.jpg/687px-Hans_Memling_-_Portrait_of_a_Man_-_WGA06755.jpg",
            "year": "1480"
        },
        {
            "title": "Portrait of a Lady",
            "artist": "Hans Memling",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Hans_Memling_-_Portrait_of_a_Lady_-_WGA06755.jpg/687px-Hans_Memling_-_Portrait_of_a_Lady_-_WGA06755.jpg",
            "year": "1480"
        },
        {
            "title": "Self-Portrait",
            "artist": "Caravaggio",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Caravaggio_-_Self-Portrait_-_WGA06755.jpg/687px-Caravaggio_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1600"
        },
        {
            "title": "Portrait of a Man",
            "artist": "Caravaggio",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Caravaggio_-_Portrait_of_a_Man_-_WGA06755.jpg/687px-Caravaggio_-_Portrait_of_a_Man_-_WGA06755.jpg",
            "year": "1600"
        },
        {
            "title": "Self-Portrait",
            "artist": "El Greco",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/El_Greco_-_Self-Portrait_-_WGA06755.jpg/687px-El_Greco_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1600"
        },
        {
            "title": "Portrait of a Man",
            "artist": "El Greco",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/El_Greco_-_Portrait_of_a_Man_-_WGA06755.jpg/687px-El_Greco_-_Portrait_of_a_Man_-_WGA06755.jpg",
            "year": "1600"
        },
        {
            "title": "Self-Portrait",
            "artist": "Peter Paul Rubens",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Peter_Paul_Rubens_-_Self-Portrait_-_WGA06755.jpg/687px-Peter_Paul_Rubens_-_Self-Portrait_-_WGA06755.jpg",
            "year": "1620"
        },
        {
            "title": "Portrait of a Lady",
            "artist": "Peter Paul Rubens",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Peter_Paul_Rubens_-_Portrait_of_a_Lady_-_WGA06755.jpg/687px-Peter_Paul_Rubens_-_Portrait_of_a_Lady_-_WGA06755.jpg",
            "year": "1620"
        }
    ]
    
    session = requests.Session()
    results = []
    downloaded_count = 0
    failed_count = 0
    
    for i, portrait in enumerate(portrait_urls):
        print(f"[{i+1}/{len(portrait_urls)}] 다운로드 중: {portrait['artist']} - {portrait['title']}")
        
        try:
            response = session.get(portrait['url'], headers=get_random_headers(), timeout=30)
            
            if response.status_code == 200 and response.content:
                # 파일명 생성
                safe_title = re.sub(r"[^\w\-\.]+", "_", portrait['title'])[:50]
                safe_artist = re.sub(r"[^\w\-\.]+", "_", portrait['artist'])[:30]
                filename = f"{i+1:03d}_{safe_artist}_{safe_title}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                
                # 이미지 저장
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                file_size_mb = len(response.content) / (1024 * 1024)
                print(f"  ✅ 다운로드 완료: {filename} ({file_size_mb:.1f}MB)")
                
                downloaded_count += 1
                results.append({
                    "object_id": f"sample_{i+1}",
                    "title": portrait['title'],
                    "artist": portrait['artist'],
                    "year": portrait['year'],
                    "filename": filename,
                    "file_size_mb": file_size_mb,
                    "status": "success",
                    "source": "public_domain"
                })
                
            else:
                print(f"  ❌ 다운로드 실패: HTTP {response.status_code}")
                failed_count += 1
                results.append({
                    "object_id": f"sample_{i+1}",
                    "title": portrait['title'],
                    "artist": portrait['artist'],
                    "year": portrait['year'],
                    "status": "failed",
                    "reason": f"HTTP {response.status_code}",
                    "source": "public_domain"
                })
                
        except Exception as e:
            print(f"  ❌ 다운로드 오류: {e}")
            failed_count += 1
            results.append({
                "object_id": f"sample_{i+1}",
                "title": portrait['title'],
                "artist": portrait['artist'],
                "year": portrait['year'],
                "status": "failed",
                "reason": str(e),
                "source": "public_domain"
            })
        
        human_delay()
    
    return results, downloaded_count, failed_count

def main():
    print("🏛️ === 대안 미술관 초상화 다운로더 ===")
    print("📋 공개 도메인 초상화 수집")
    print("=" * 60)
    
    # 샘플 초상화 다운로드 (API 키 없이)
    results, downloaded_count, failed_count = download_sample_portraits()
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(METADATA_DIR, f"alternative_portraits_{timestamp}.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 최종 통계
    total_size = sum([r.get('file_size_mb', 0) for r in results if r.get('status') == 'success'])
    
    print(f"\n🎉 === 대안 초상화 수집 완료 ===")
    print(f"✅ 총 다운로드: {downloaded_count}개")
    print(f"❌ 다운로드 실패: {failed_count}개")
    print(f"📁 이미지 저장: {SAVE_DIR}")
    print(f"📄 메타데이터: {csv_path}")
    print(f"💾 총 파일 크기: {total_size:.1f}MB")
    
    print(f"\n💡 팁: 더 많은 초상화를 원하시면:")
    print(f"   1. Rijksmuseum API 키 발급: https://www.rijksmuseum.nl/en/api")
    print(f"   2. Europeana API 키 발급: https://pro.europeana.eu/get-api")
    print(f"   3. 스크립트의 API 키를 업데이트하세요!")

if __name__ == "__main__":
    main()
