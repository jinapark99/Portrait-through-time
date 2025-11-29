# 📦 GitHub 업로드 가이드

## ✅ 올려야 할 파일들 (필수)

### 핵심 파일들
- ✅ `portrait.py` - LoRA 학습 스크립트
- ✅ `streamlit_portrait_web.py` - 웹 앱 메인 파일
- ✅ `selfie_to_portrait.py` - 셀피→초상화 변환 스크립트
- ✅ `download_images_fast.py` - 데이터셋 다운로드 스크립트
- ✅ `README.md` - 프로젝트 설명서
- ✅ `requirements.txt` - 필요한 패키지 목록
- ✅ `.gitignore` - 업로드 제외 파일 목록

### 선택적 파일들 (있으면 좋음)
- `analyze_emotions.py` - 감정 분석 스크립트
- `emotion_style_portrait.py` - 감정 기반 스타일 변환
- `create_portrait.py` - 초상화 생성 유틸리티
- `filter_portraits.py` - 데이터 필터링 스크립트

## ❌ 올리지 말아야 할 파일들

### 자동으로 제외됨 (.gitignore에 의해)
- ❌ `lora_trained_model/` - 학습된 모델 (너무 큼)
- ❌ `data/` - 이미지 데이터셋 (너무 큼)
- ❌ `venv/` - 가상환경
- ❌ `*.png`, `*.jpg` - 생성된 이미지들
- ❌ `*.csv` - 대용량 데이터 파일
- ❌ `*.log`, `*.txt` - 로그 파일들
- ❌ `checkpoint-*/` - 체크포인트 폴더들

## 🚀 업로드 방법

1. **GitHub에서 저장소 생성**
   - GitHub.com 접속 → New repository
   - 이름: `portrait-through-time` (또는 원하는 이름)
   - Public 선택

2. **터미널에서 실행** (Portrait 폴더에서)
```bash
# Git 초기화
git init

# 모든 파일 추가 (gitignore에 따라 자동 제외됨)
git add .

# 커밋
git commit -m "Initial commit: Portrait Through Time project"

# GitHub 저장소 연결 (YOUR_USERNAME과 YOUR_REPO_NAME 변경!)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 업로드
git branch -M main
git push -u origin main
```

## 💡 팁

- 모델 파일은 GitHub에 올리지 말고 Hugging Face Hub나 Google Drive에 올리세요
- README.md에 모델 다운로드 링크를 추가하세요
- 예시 이미지는 `example.png` 같은 이름으로 저장하면 올라갑니다

