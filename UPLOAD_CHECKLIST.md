# ✅ GitHub 업로드 체크리스트

## 📋 올라갈 파일 목록 (자동으로 선택됨)

### ✅ 핵심 파일들
- `portrait.py` - LoRA 학습 스크립트 ⭐
- `streamlit_portrait_web.py` - 웹 앱 메인 파일 ⭐
- `selfie_to_portrait.py` - 셀피→초상화 변환 ⭐
- `download_images_fast.py` - 데이터셋 다운로드 ⭐
- `README.md` - 프로젝트 설명서 ⭐
- `requirements.txt` - 패키지 목록 ⭐
- `.gitignore` - 제외 파일 목록 ⭐

### ✅ 추가 스크립트들
- `analyze_emotions.py`
- `emotion_style_portrait.py`
- `create_portrait.py`
- `download_*.py` (다양한 다운로드 스크립트들)
- `test_lora_loading.py`
- 기타 유틸리티 스크립트들

## ❌ 자동으로 제외되는 파일들 (.gitignore에 의해)

- ❌ `lora_trained_model/` - 학습된 모델 (너무 큼)
- ❌ `data/` - 이미지 데이터셋
- ❌ `venv/` - 가상환경
- ❌ `*.png`, `*.jpg` - 생성된 이미지들
- ❌ `*.csv` - 대용량 데이터 파일
- ❌ `*.zip` - 압축 파일
- ❌ `checkpoint-*/` - 체크포인트 폴더들

## 🚀 업로드 단계

### 1단계: GitHub에서 저장소 만들기
1. https://github.com 접속
2. 우측 상단 **+** 클릭 → **New repository**
3. Repository name: `portrait-through-time` (또는 원하는 이름)
4. Description: "AI-powered portrait generation using emotion detection and classical art styles"
5. **Public** 선택
6. **"Initialize this repository with a README"** 체크 해제
7. **Create repository** 클릭

### 2단계: 터미널에서 실행

Portrait 폴더에서 다음 명령어들을 순서대로 실행하세요:

```bash
# 1. 현재 위치 확인
cd /Users/jinapark/Desktop/py4us/Portrait

# 2. Git 초기화 (이미 했으면 스킵)
git init

# 3. 모든 파일 추가 (gitignore에 따라 자동 제외됨)
git add .

# 4. 어떤 파일들이 추가되었는지 확인
git status

# 5. 첫 커밋
git commit -m "Initial commit: Portrait Through Time project"

# 6. GitHub 저장소 연결
# ⚠️ YOUR_USERNAME과 YOUR_REPO_NAME을 실제 값으로 변경하세요!
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 7. 메인 브랜치로 이름 변경
git branch -M main

# 8. GitHub에 업로드
git push -u origin main
```

### 3단계: 확인
- GitHub 저장소 페이지에서 파일들이 올라갔는지 확인
- README.md가 제대로 보이는지 확인

## 💡 주의사항

1. **모델 파일은 올라가지 않습니다** (너무 크기 때문)
   - 모델이 필요하면 Hugging Face Hub나 Google Drive에 별도 업로드
   - README에 다운로드 링크 추가

2. **첫 업로드 시 GitHub 로그인 필요**
   - Personal Access Token 필요할 수 있음

3. **저장소 URL을 README에 추가**
   - 업로드 후 저장소 URL을 README.md에 추가하세요

