import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from insightface.app import FaceAnalysis
import random

# Label Map
LABEL_MAP = {
    "前田敦子": "bdgh", "坂井真紀": "bdgh", "森川葵": "bdgi", "真木よう子": "bdfh",
    "蓮佛美沙子": "bdgi", "ソニン": "bdfh", "山田杏奈": "befh", "新垣結衣": "befh",
    "大原麗子": "bdfi", "玉城ティナ": "befi", "南沙良": "adfh", "桜田ひより": "befi",
    "長谷川京子": "adfh", "桜庭ななみ": "adfi", "瀧本美織": "bdfi", "木南晴夏": "adgh",
    "りょう": "aegh", "臼田あさ美": "aefh", "井川遥": "begh", "米倉涼子": "aefh",
    "中谷美紀": "adgh", "木村文乃": "begh", "内田理央": "aegh", "吉高由里子": "begi",
    "中条あやみ": "adfi", "薬師丸ひろ子": "begi"
}

def load_insightface():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

def get_3d_landmarks(app, img_path):
    try:
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: return None, None
        faces = app.get(img)
        if not faces: return None, None
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        if face.landmark_3d_68 is not None:
            return face.landmark_3d_68, img
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def rigid_transform_3d(A, B):
    """
    Aligns A to B based on eyes.
    1. Calculate Left/Right Eye Centers for A and B.
    2. Scale A so that Eye Distance matches B.
    3. Translate A so that mid-eye point matches B.
    4. (Optional) Rotate A so eyes are horizontal if needed (assuming B is target).
    """
    # 36-41: Left Eye, 42-47: Right Eye (0-indexed)
    l_eye_idx = list(range(36, 42))
    r_eye_idx = list(range(42, 48))
    
    def get_eye_centers(pts):
        le = np.mean(pts[l_eye_idx], axis=0)
        re = np.mean(pts[r_eye_idx], axis=0)
        return le, re

    le_A, re_A = get_eye_centers(A)
    le_B, re_B = get_eye_centers(B)
    
    # Distance
    dist_A = np.linalg.norm(le_A - re_A)
    dist_B = np.linalg.norm(le_B - re_B)
    
    # Scale A
    scale = dist_B / (dist_A + 1e-6)
    A_scaled = A * scale
    
    # Re-calculate A centers after scale
    le_A, re_A = get_eye_centers(A_scaled)
    mid_A = (le_A + re_A) / 2
    mid_B = (le_B + re_B) / 2
    
    # Translate A to match mid-point
    translation = mid_B - mid_A
    A_aligned = A_scaled + translation
    
    # Rotation (align eye vector)
    # Vector A eyes
    vec_A = re_A - le_A
    vec_A = vec_A / np.linalg.norm(vec_A)
    
    # Vector B eyes
    vec_B = re_B - le_B
    vec_B = vec_B / np.linalg.norm(vec_B)
    
    # Rotation matrix to align vec_A to vec_B
    # v = a x b, s = ||v||, c = a . b
    v = np.cross(vec_A, vec_B)
    s = np.linalg.norm(v)
    c = np.dot(vec_A, vec_B)
    
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
        
        # Rotate around mid_point (mid_B is now A's center too)
        A_aligned = np.dot(A_aligned - mid_B, R.T) + mid_B

    return A_aligned

def plot_overlapping_views(lm1, name1, lm2, name2, pair_idx, out_dir):
    # Align lm1 to lm2 (Target is lm2)
    # Use eyes and nose for stable alignment (indices depend on 68 point scheme)
    # 36-41 (Right Eye), 42-47 (Left Eye), 27-35 (Nose)
    # Rough indices: 36:48 (Eyes), 27:36 (Nose)
    
    # Simple rigorous alignment using ALL points for shape comparison
    lm1_aligned = rigid_transform_3d(lm1, lm2)
    
    # Center both at 0
    center = np.mean(lm2, axis=0)
    lm2_centered = lm2 - center
    lm1_centered = lm1_aligned - center

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Overlay Comparison: Blue={name1}(a) vs Red={name2}(b)", fontsize=16)

    # View 1: Front (XY Projection) via 3D plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(lm1_centered[:,0], lm1_centered[:,1], lm1_centered[:,2], c='blue', marker='o', s=20, alpha=0.6, label=name1)
    ax1.scatter(lm2_centered[:,0], lm2_centered[:,1], lm2_centered[:,2], c='red', marker='^', s=20, alpha=0.6, label=name2)
    ax1.view_init(elev=-90, azim=-90) # Front view (assuming standard InsightFace Z-forward)
    ax1.set_title("Front View")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # View 2: Side (YZ Projection)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(lm1_centered[:,0], lm1_centered[:,1], lm1_centered[:,2], c='blue', marker='o', s=20, alpha=0.6)
    ax2.scatter(lm2_centered[:,0], lm2_centered[:,1], lm2_centered[:,2], c='red', marker='^', s=20, alpha=0.6)
    ax2.view_init(elev=0, azim=0) # Side view
    ax2.set_title("Side View (Profile)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # View 3: Top (XZ Projection)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(lm1_centered[:,0], lm1_centered[:,1], lm1_centered[:,2], c='blue', marker='o', s=20, alpha=0.6)
    ax3.scatter(lm2_centered[:,0], lm2_centered[:,1], lm2_centered[:,2], c='red', marker='^', s=20, alpha=0.6)
    ax3.view_init(elev=-90, azim=0) # Top view
    ax3.set_title("Top View")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    # Draw lines connecting same landmarks to visualize diff
    # Only draw lines if distance is significant to avoid clutter
    for i in range(len(lm1_centered)):
        dist = np.linalg.norm(lm1_centered[i] - lm2_centered[i])
        if dist > 5.0: # Threshold
             # Use max of coordinates to place lines? No, draw in all subplots?
             # Matplotlib 3D lines are tricky in shared views. 
             # Skip lines for clarity, dots overlap shows the diff.
             pass

    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"pair_{pair_idx}_overlay.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def get_label(name):
    code = LABEL_MAP.get(name, "????")
    if 'a' in code: return 'a'
    if 'b' in code: return 'b'
    return '?'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="master_data", help="Source directory")
    parser.add_argument("--out_dir", type=str, default="outputs/3d_comparison", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Group
    group_a, group_b = [], []
    for name, code in LABEL_MAP.items():
        person_dir = None
        # Deep search for person dir
        for root, dirs, files in os.walk(args.data_dir):
            if os.path.basename(root) == name:
                person_dir = root
                break
        if not person_dir: continue

        images = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not images: continue
        
        if 'a' in code: group_a.append((name, images))
        elif 'b' in code: group_b.append((name, images))

    if not group_a or not group_b:
        print("Not enough data.")
        return

    app = load_insightface()
    
    # Compare 3 pairs
    for i in range(3):
        person_a = random.choice(group_a)
        person_b = random.choice(group_b)
        
        # Try to find valid landmarks (retry if fail)
        lm_a, lm_b = None, None
        
        # Try up to 5 times to find a good image for A
        for _ in range(5):
            path_a = random.choice(person_a[1])
            lm_a, _ = get_3d_landmarks(app, path_a)
            if lm_a is not None: break
            
        # Try up to 5 times to find a good image for B
        for _ in range(5):
            path_b = random.choice(person_b[1])
            lm_b, _ = get_3d_landmarks(app, path_b)
            if lm_b is not None: break

        if lm_a is not None and lm_b is not None:
            print(f"Overlaying Pair {i+1}: {person_a[0]}(a) vs {person_b[0]}(b)")
            plot_overlapping_views(lm_a, person_a[0], lm_b, person_b[0], i+1, args.out_dir)
        else:
            print(f"Skipping Pair {i+1} due to detection failure.")

if __name__ == "__main__":
    main()
