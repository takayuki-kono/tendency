import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os

# --- Config ---
TEST_DIR = "master_data" # Change this if you want to test specific dir
OUTPUT_DIR = "debug_landmarks"

def imread_safe(path):
    try:
        # 日本語パス対応のため、numpy経由で読み込んでdecodeする
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def main():
    # part2b_filter.pyと同じ設定
    print("Loading InsightFace model...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    print("Model loaded.")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find a few images to test
    image_paths = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= 50: break
        if len(image_paths) >= 50: break
    
    if not image_paths:
        print("No images found in master_data to test.")
        return

    print(f"Testing on {len(image_paths)} images...")

    success_count = 0
    for img_path in image_paths:
        img = imread_safe(img_path)
        if img is None: 
            print(f"Could not read: {img_path}")
            continue
        
        # 小さすぎる画像はスキップ
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            print(f"Skipping small image ({w}x{h}): {img_path}")
            continue
        
        faces = app.get(img)
        
        if not faces:
            print(f"No face found in {img_path}")
            continue
            
        face = faces[0]
        lmk = face.landmark_2d_106
        lmk3d = face.landmark_3d_68
        
        # Draw 106 2D landmarks (Green small dots)
        if lmk is not None:
             for i in range(106):
                pt = (int(lmk[i][0]), int(lmk[i][1]))
                cv2.circle(img, pt, 1, (0, 255, 0), -1)
        
        # Draw 3D landmarks (Red large dots) projected to 2D
        # Note: landmark_3d_68 xy coordinates are usually aligned to image space
        if lmk3d is not None:
            # All 68 points with index numbers
            for i in range(68):
                pt = (int(lmk3d[i][0]), int(lmk3d[i][1]))
                cv2.circle(img, pt, 2, (0, 0, 255), -1)
                cv2.putText(img, str(i), (pt[0]+3, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Highlight Cheek Points (3 and 13)
            # Index 3: Right cheek (viewer's right?) -> Standard 68 point: 0-16 are jawline.
            # 0 is left ear, 16 is right ear. 3 is left cheek, 13 is right cheek (or vice versa depending on definition)
            
            # Let's highlight specific indices we used
            idx_a = 3
            idx_b = 13
            
            pt_a = (int(lmk3d[idx_a][0]), int(lmk3d[idx_a][1]))
            pt_b = (int(lmk3d[idx_b][0]), int(lmk3d[idx_b][1]))
            
            cv2.circle(img, pt_a, 5, (255, 0, 0), -1) # Blue: Index 3
            cv2.putText(img, "3", pt_a, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            cv2.circle(img, pt_b, 5, (255, 255, 0), -1) # Cyan: Index 13
            cv2.putText(img, "13", pt_b, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Save
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"debug_{filename}")
        cv2.imwrite(save_path, img)
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
