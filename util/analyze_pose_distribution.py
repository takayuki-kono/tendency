import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

def collect_pose_values(root_dir, max_images=1000):
    """ディレクトリから画像のpose値を収集"""
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    pitch_values = []
    yaw_values = []
    
    # 画像ファイルを収集
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, f))
                if len(image_files) >= max_images:
                    break
        if len(image_files) >= max_images:
            break
    
    print(f"Found {len(image_files)} images. Analyzing pose values...")
    
    for img_path in tqdm(image_files):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            faces = app.get(img)
            if not faces:
                continue
            
            face = faces[0]
            if face.pose is not None:
                pitch, yaw, roll = face.pose
                pitch_values.append(abs(pitch))
                yaw_values.append(abs(yaw))
        except:
            continue
    
    return np.array(pitch_values), np.array(yaw_values)

def analyze_distribution(values, name):
    """分布を分析して表示"""
    print(f"\n{'='*50}")
    print(f"{name} の分布:")
    print(f"{'='*50}")
    print(f"サンプル数: {len(values)}")
    print(f"平均: {np.mean(values):.2f}°")
    print(f"中央値: {np.median(values):.2f}°")
    print(f"標準偏差: {np.std(values):.2f}°")
    print(f"\nパーセンタイル:")
    for p in [50, 60, 70, 75, 80, 85, 90, 95]:
        threshold = np.percentile(values, p)
        print(f"  上位{100-p:2d}% (>= {threshold:5.2f}°) をフィルタ")

# メイン処理
print("Pitch/Yaw分布解析ツール")
print("=" * 50)

# trainディレクトリを解析
root_dir = "train"
if not os.path.exists(root_dir):
    print(f"Error: {root_dir} が見つかりません")
else:
    pitch_vals, yaw_vals = collect_pose_values(root_dir, max_images=1000)
    
    if len(pitch_vals) > 0:
        analyze_distribution(pitch_vals, "Pitch (前傾後傾)")
        analyze_distribution(yaw_vals, "Yaw (横向き)")
        
        print(f"\n{'='*50}")
        print("推奨設定:")
        print(f"{'='*50}")
        
        # 現在の閾値での除外率を計算
        current_pitch = 15
        current_yaw = 15
        pitch_filtered = np.sum(pitch_vals > current_pitch) / len(pitch_vals) * 100
        yaw_filtered = np.sum(yaw_vals > current_yaw) / len(yaw_vals) * 100
        
        print(f"\n現在の設定 (abs(pitch) > {current_pitch}, abs(yaw) > {current_yaw}):")
        print(f"  Pitch: 上位 {pitch_filtered:.1f}% をフィルタ")
        print(f"  Yaw:   上位 {yaw_filtered:.1f}% をフィルタ")
    else:
        print("pose値を取得できませんでした")
