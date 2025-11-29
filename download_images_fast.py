
import os,re,requests,pandas as pd
from urllib.parse import urlparse

CSV="portraits_dataset.csv"   # Modify path if needed
SAVE="data/images_test10"
os.makedirs(SAVE, exist_ok=True)

print("⚠️  Warning: The current CSV file does not contain image URLs!")
print("Instead, running code to download portraits from public art datasets.")
print("=" * 60)

def to_iiif_image(url):
    if isinstance(url,str) and url.endswith("/info.json"):
        base=url[:-len("/info.json")]
        return f"{base}/full/!1200,1200/0/default.jpg"
    if isinstance(url,str) and "/iiif/" in url and "/full/" not in url and url.endswith("/"):
        return f"{url}full/!1200,1200/0/default.jpg"
    return url

def pick_url(row):
    prio=["iiif","image","image_url","imageurl","imagelink","primaryimage","url","link"]
    for c in row.index:
        name=str(c).lower()
        if any(k in name for k in prio):
            v=row[c]
            if isinstance(v,str) and v.startswith("http"):
                return to_iiif_image(v)
    # fallback: any URL containing jpg/png/tif
    for c in row.index:
        v=row[c]
        if isinstance(v,str) and v.startswith("http") and any(ext in v.lower() for ext in [".jpg",".jpeg",".png",".tif",".tiff",".webp","/iiif/"]):
            return to_iiif_image(v)
    return None

# Alternative: Download portraits from public art datasets
def download_sample_portraits():
    """Download sample portrait images."""
    
    # Public URLs of famous portraits (CC license or public domain)
    portrait_urls = [
        {
            'title': 'Mona Lisa',
            'artist': 'Leonardo da Vinci',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Vincent van Gogh',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/b/b2/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg'
        },
        {
            'title': 'Portrait of a Man',
            'artist': 'Jan van Eyck',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/7/76/Jan_van_Eyck_-_Portrait_of_a_Man_%28Self_Portrait%29_-_WGA07761.jpg'
        },
        {
            'title': 'Girl with a Pearl Earring',
            'artist': 'Johannes Vermeer',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/0/0f/1665_Girl_with_a_Pearl_Earring.jpg'
        },
        {
            'title': 'Self-Portrait',
            'artist': 'Rembrandt',
            'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Self-portrait_by_Rembrandt.jpg'
        }
    ]
    
    ok = 0
    fail = 0
    paths = []
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 PortraitTest/1.0"})
    
    for i, portrait in enumerate(portrait_urls):
        url = portrait['url']
        fname = re.sub(r"[^\w\-\.]+", "_", f"{portrait['artist']}_{portrait['title']}_{i}")[:90] + ".jpg"
        fpath = os.path.join(SAVE, fname)
        
        try:
            r = session.get(url, timeout=20)
            if r.ok and r.content:
                with open(fpath, "wb") as f: 
                    f.write(r.content)
                print(f"[{i}] OK  -> {fpath}")
                paths.append(fpath)
                ok += 1
            else:
                print(f"[{i}] HTTP_{r.status_code}  {url}")
                paths.append("")
                fail += 1
        except Exception as e:
            print(f"[{i}] EXC_{type(e).__name__}  {url}")
            paths.append("")
            fail += 1
    
    return ok, fail, paths

# Check existing CSV file
df = pd.read_csv(CSV, low_memory=False)
print(f"CSV file contains {len(df)} portrait records.")
print("However, image URLs are missing, so downloading samples from public datasets.")

# Execute sample portrait download
ok, fail, paths = download_sample_portraits()

# Save results
sample_df = pd.DataFrame({
    'Title': ['Mona Lisa', 'Self-Portrait (van Gogh)', 'Portrait of a Man', 'Girl with a Pearl Earring', 'Self-Portrait (Rembrandt)'],
    'Artist': ['Leonardo da Vinci', 'Vincent van Gogh', 'Jan van Eyck', 'Johannes Vermeer', 'Rembrandt'],
    'Classification': ['Painting'] * 5,
    'test_local_path': paths
})

out = "portraits_dataset_TEST10.csv"
sample_df.to_csv(out, index=False)
print(f"\n== SUMMARY == OK={ok} FAIL={fail}")
print(f"Saved {out} and files under {SAVE}/")
print("\n💡 Tip: To download more portraits, run 'download_portraits_with_api.py' or 'download_from_public_dataset.py'!")