import os
import hashlib
import shutil
from collections import defaultdict

def get_file_hash(filepath):
    """파일의 MD5 해시를 계산합니다."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def remove_duplicate_images(folder_path):
    """중복된 이미지 파일들을 제거합니다."""
    
    if not os.path.exists(folder_path):
        print(f"폴더가 존재하지 않습니다: {folder_path}")
        return
    
    print(f"🔍 {folder_path}에서 중복 이미지를 검사합니다...")
    
    # 파일 해시별로 그룹화
    hash_groups = defaultdict(list)
    file_hashes = {}
    
    # 모든 이미지 파일의 해시 계산
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    
    print(f"📁 총 {len(image_files)}개의 이미지 파일을 검사합니다...")
    
    for filename in image_files:
        filepath = os.path.join(folder_path, filename)
        file_hash = get_file_hash(filepath)
        
        if file_hash:
            hash_groups[file_hash].append(filename)
            file_hashes[filename] = file_hash
            print(f"  ✓ {filename} - 해시: {file_hash[:8]}...")
    
    # 중복 파일 찾기
    duplicates = {hash_val: files for hash_val, files in hash_groups.items() if len(files) > 1}
    
    if not duplicates:
        print("🎉 중복 파일이 없습니다!")
        return
    
    print(f"\n🔍 {len(duplicates)}개의 중복 그룹을 발견했습니다:")
    
    total_duplicates = 0
    removed_count = 0
    
    # 백업 폴더 생성
    backup_folder = os.path.join(folder_path, "duplicates_backup")
    os.makedirs(backup_folder, exist_ok=True)
    
    for hash_val, files in duplicates.items():
        print(f"\n📊 해시 {hash_val[:8]}... (중복 {len(files)}개):")
        for i, filename in enumerate(files):
            print(f"  {i+1}. {filename}")
        
        # 첫 번째 파일은 유지, 나머지는 백업으로 이동
        keep_file = files[0]
        duplicate_files = files[1:]
        
        print(f"  ✅ 유지: {keep_file}")
        
        for dup_file in duplicate_files:
            source_path = os.path.join(folder_path, dup_file)
            backup_path = os.path.join(backup_folder, dup_file)
            
            try:
                shutil.move(source_path, backup_path)
                print(f"  🗑️ 백업으로 이동: {dup_file}")
                removed_count += 1
            except Exception as e:
                print(f"  ❌ 이동 실패: {dup_file} - {e}")
        
        total_duplicates += len(duplicate_files)
    
    print(f"\n🎉 === 중복 제거 완료 ===")
    print(f"✅ 유지된 파일: {len(image_files) - total_duplicates}개")
    print(f"🗑️ 제거된 중복: {removed_count}개")
    print(f"📁 백업 위치: {backup_folder}")
    
    return removed_count

def analyze_remaining_images(folder_path):
    """남은 이미지들을 분석합니다."""
    
    if not os.path.exists(folder_path):
        return
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    
    print(f"\n📊 === 남은 이미지 분석 ===")
    print(f"📁 총 이미지: {len(image_files)}개")
    
    # 파일 크기별 분석
    size_groups = {
        'small (< 1MB)': 0,
        'medium (1-5MB)': 0,
        'large (> 5MB)': 0
    }
    
    total_size = 0
    
    for filename in image_files:
        filepath = os.path.join(folder_path, filename)
        try:
            size = os.path.getsize(filepath)
            total_size += size
            
            if size < 1024 * 1024:  # < 1MB
                size_groups['small (< 1MB)'] += 1
            elif size < 5 * 1024 * 1024:  # < 5MB
                size_groups['medium (1-5MB)'] += 1
            else:  # >= 5MB
                size_groups['large (> 5MB)'] += 1
                
        except:
            pass
    
    print(f"\n📏 파일 크기별 분포:")
    for size_group, count in size_groups.items():
        print(f"  • {size_group}: {count}개")
    
    print(f"\n💾 총 용량: {total_size / (1024*1024):.1f}MB")

def main():
    print("🧹 중복 이미지 제거 도구")
    print("=" * 50)
    
    # CSV 포트레이트 폴더에서 중복 제거
    csv_folder = "data/csv_portraits"
    
    if os.path.exists(csv_folder):
        removed_count = remove_duplicate_images(csv_folder)
        analyze_remaining_images(csv_folder)
        
        if removed_count > 0:
            print(f"\n💡 팁:")
            print(f"• 중복된 파일들은 {csv_folder}/duplicates_backup/ 폴더로 백업되었습니다")
            print(f"• 필요하면 언제든지 복원할 수 있습니다")
    else:
        print(f"❌ 폴더가 존재하지 않습니다: {csv_folder}")

if __name__ == "__main__":
    main()
