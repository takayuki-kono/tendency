import os
import cv2
import random
from insightface.app import FaceAnalysis

# InsightFace初期化
face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320))

# preprocessed_multitaskから20枚サンプリング
data_dir = "preprocessed_multitask/train"
all_images = []

for root, _, files in os.walk(data_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            all_images.append(os.path.join(root, f))

# ランダムに20枚選択
sample_images = random.sample(all_images, min(20, len(all_images)))

print("=" * 60)
print("Face Size Ratio Analysis (20 samples)")
print("=" * 60)

ratios = []
for img_path in sample_images:
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    faces = face_app.get(img)

    if faces:
        face = faces[0]
        x1, y1, x2, y2 = face.bbox

        face_area = (x2 - x1) * (y2 - y1)
        img_area = h * w
        size_ratio = face_area / img_area

        ratios.append(size_ratio)

        filename = os.path.basename(img_path)
        print(f"{filename:30s} | {size_ratio:.3f} ({size_ratio*100:5.1f}%)")
    else:
        print(f"{os.path.basename(img_path):30s} | No face detected")

if ratios:
    print("\n" + "=" * 60)
    print(f"Statistics:")
    print(f"  Min:  {min(ratios):.3f} ({min(ratios)*100:.1f}%)")
    print(f"  Max:  {max(ratios):.3f} ({max(ratios)*100:.1f}%)")
    print(f"  Mean: {sum(ratios)/len(ratios):.3f} ({sum(ratios)/len(ratios)*100:.1f}%)")
    print("=" * 60)

    # 80%超の画像数
    over_80 = sum(1 for r in ratios if r > 0.8)
    print(f"\nImages with size ratio > 80%: {over_80}/{len(ratios)}")
