"""
pHash ベースの類似画像検出テスト
削除はせず、検出結果のみ表示
"""
import os
import sys
import argparse
from PIL import Image
from collections import defaultdict

try:
    import imagehash
except ImportError:
    print("imagehash ライブラリが必要です。")
    print("pip install imagehash")
    sys.exit(1)

def get_phash(img_path, hash_size=16, resize_to=None):
    """画像からpHashを計算"""
    try:
        img = Image.open(img_path).convert('L')
        # 強制リサイズ（解像度差を吸収するため）
        if resize_to:
            img = img.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
        return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

def get_dhash(img_path, hash_size=16, resize_to=None):
    """画像からdHashを計算（解像度変化に強い）"""
    try:
        img = Image.open(img_path).convert('L')
        if resize_to:
            img = img.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
        return imagehash.dhash(img, hash_size=hash_size)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

def find_duplicates(folder, threshold=10, hash_size=16, resize_to=None, use_dhash=False):
    """
    フォルダ内の画像から類似画像ペアを検出
    threshold: Hamming distance の閾値（小さいほど厳格）
    resize_to: 画像を強制リサイズするサイズ（Noneなら元サイズ）
    use_dhash: Trueならdhashを使用（解像度変化に強い）
    """
    # 画像ファイル収集
    image_files = []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(os.path.join(folder, f))
    
    print(f"Found {len(image_files)} images in {folder}")
    print(f"Options: threshold={threshold}, hash_size={hash_size}, resize_to={resize_to}, use_dhash={use_dhash}")
    
    if len(image_files) < 2:
        print("Not enough images to compare.")
        return []
    
    # ハッシュ計算
    hashes = {}
    hash_func = get_dhash if use_dhash else get_phash
    print(f"Calculating {'dHash' if use_dhash else 'pHash'}...")
    
    for i, img_path in enumerate(image_files):
        h = hash_func(img_path, hash_size, resize_to)
        if h is not None:
            hashes[img_path] = h
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(image_files)}")
    
    print(f"Calculated hashes for {len(hashes)} images.")
    
    # ペア比較
    print(f"Comparing pairs (threshold={threshold})...")
    duplicates = []
    paths = list(hashes.keys())
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            p1, p2 = paths[i], paths[j]
            distance = hashes[p1] - hashes[p2]
            if distance <= threshold:
                duplicates.append((p1, p2, distance))
    
    return duplicates

def group_duplicates(duplicates):
    """重複ペアをグループ化"""
    # Union-Find的なグループ化
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for p1, p2, _ in duplicates:
        union(p1, p2)
    
    groups = defaultdict(list)
    all_paths = set()
    for p1, p2, _ in duplicates:
        all_paths.add(p1)
        all_paths.add(p2)
    
    for path in all_paths:
        groups[find(path)].append(path)
    
    return list(groups.values())

def main():
    parser = argparse.ArgumentParser(description="pHash/dHash Duplicate Detection Test")
    parser.add_argument("folder", type=str, help="Target folder")
    parser.add_argument("--threshold", type=int, default=10, help="Hamming distance threshold (default: 10)")
    parser.add_argument("--hash_size", type=int, default=16, help="Hash size (default: 16)")
    parser.add_argument("--resize", type=int, default=None, help="Force resize images to NxN before hashing (e.g., 64)")
    parser.add_argument("--use_dhash", action="store_true", help="Use dHash instead of pHash")
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Folder not found: {args.folder}")
        return
    
    duplicates = find_duplicates(
        args.folder, 
        args.threshold, 
        args.hash_size,
        resize_to=args.resize,
        use_dhash=args.use_dhash
    )
    
    if not duplicates:
        print("\nNo duplicates found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(duplicates)} duplicate pairs:")
    print(f"{'='*60}")
    
    for p1, p2, dist in duplicates[:20]:  # 最大20ペア表示
        size1 = os.path.getsize(p1)
        size2 = os.path.getsize(p2)
        print(f"\n  Distance: {dist}")
        print(f"    {os.path.basename(p1)} ({size1:,} bytes)")
        print(f"    {os.path.basename(p2)} ({size2:,} bytes)")
        if size1 > size2:
            print(f"    -> Keep: {os.path.basename(p1)}")
        else:
            print(f"    -> Keep: {os.path.basename(p2)}")
    
    if len(duplicates) > 20:
        print(f"\n  ... and {len(duplicates) - 20} more pairs")
    
    # グループ化表示
    groups = group_duplicates(duplicates)
    print(f"\n{'='*60}")
    print(f"Grouped into {len(groups)} duplicate groups:")
    print(f"{'='*60}")
    
    for i, group in enumerate(groups[:10], 1):
        print(f"\nGroup {i} ({len(group)} images):")
        # サイズ順にソート（大きい順）
        sorted_group = sorted(group, key=lambda x: os.path.getsize(x), reverse=True)
        for j, path in enumerate(sorted_group):
            size = os.path.getsize(path)
            marker = " [KEEP]" if j == 0 else " [DELETE]"
            print(f"    {os.path.basename(path)} ({size:,} bytes){marker}")
    
    if len(groups) > 10:
        print(f"\n  ... and {len(groups) - 10} more groups")

if __name__ == "__main__":
    main()
