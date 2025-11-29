# filter_portraits.py
import re, pandas as pd, sys

src = sys.argv[1] if len(sys.argv) > 1 else "objects.csv"
df = pd.read_csv(src, low_memory=False)

def pick(colnames, *cands):
    # cand 들 중 이름이 포함된 첫 컬럼 반환 (대소문자 무시)
    for c in cands:
        for name in colnames:
            if re.search(rf"\b{c}\b", str(name), flags=re.I):
                return name
    return None

cols = df.columns
col_title   = pick(cols, "title", "object title")
col_artist  = pick(cols, "artist", "creator", "constituent", "maker", "name")
col_class   = pick(cols, "classification", "work type", "object type")
col_subj    = pick(cols, "subject", "keywords", "tags")
col_year    = pick(cols, "date", "year")
col_medium  = pick(cols, "medium", "materials")
col_img     = pick(cols, "image", "iiif", "image_url", "image link", "imageurl")

# 문자열 컬럼만 임시로 소문자화
def s(x): 
    return x.astype(str).str.lower() if x is not None else None

title_s = s(df[col_title]) if col_title else None
class_s = s(df[col_class]) if col_class else None
subj_s  = s(df[col_subj])  if col_subj  else None

# 'portrait' / 'self-portrait' 키워드로 다층 필터
mask = pd.Series([False]*len(df))
for series in [title_s, class_s, subj_s]:
    if series is not None:
        mask |= series.str.contains(r"\b(self[- ]?)?portrait\b", na=False)

portraits = df[mask].copy()

keep_cols = [c for c in [col_title, col_artist, col_year, col_medium, col_class, col_img] if c]
portraits = portraits.loc[:, dict.fromkeys(keep_cols).keys()]  # 중복 제거 순서 유지
portraits.rename(columns={
    col_title:"Title", col_artist:"Artist", col_year:"Year",
    col_medium:"Medium", col_class:"Classification", col_img:"ImageURL"
}, inplace=True)

portraits.to_csv("portraits_dataset.csv", index=False)
print(f"Saved portraits_dataset.csv with {len(portraits)} rows.")
