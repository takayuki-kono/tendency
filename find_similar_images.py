import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp

# 設定
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
BLACK_MAX = 85
GRAY_MIN = 86
GRAY_MAX = 170
WHITE_MIN = 171
COLOR_RATIO_THRESHOLD = 0.045
TILT_THRESHOLD = 0.116
OUTPUT_CSV = 'similar_images_color_tilt_split.csv'
TOP_N = 100
IMG_SIZE = 112
TRIANGLE_LANDMARK_INDICES = [(33, 263), (263, 1), (1, 33)]

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_landmarks(img_path):
    """画像から顔ランドマークを抽出（2D座標のみ）"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not loaded")
        img_rgb = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = [(l.x * IMG_SIZE, l.y * IMG_SIZE) for l in landmarks]
            nose_center = coords[1]
            coords = [(x - nose_center[0], y - nose_center[1]) for x, y in coords]
            return np.array(coords)
        else:
            print(f"No landmarks detected in {img_path}")
            return None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def compute_tilt(landmarks, indices):
    """2点間の勾配を計算（dy/dx）"""
    try:
        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) < 1e-10:
            return float('inf')
        tilt = dy / dx
        return tilt
    except Exception as e:
        print(f"Error computing tilt: {e}")
        return float('inf')

def compute_color_ratios(img_path):
    """画像を縦に2分割し、左半分と右半分の黒・灰・白の割合を計算"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not loaded")
        # 縦に2分割（左：0-55、右：56-111）
        left_half = img[:, :IMG_SIZE//2]
        right_half = img[:, IMG_SIZE//2:]
        # ピクセル数
        total_pixels = left_half.size  # 112x56=6272
        # 左半分
        left_black_pixels = np.sum(left_half <= BLACK_MAX)
        left_gray_pixels = np.sum((left_half >= GRAY_MIN) & (left_half <= GRAY_MAX))
        left_white_pixels = np.sum(left_half >= WHITE_MIN)
        left_black_ratio = left_black_pixels / total_pixels
        left_gray_ratio = left_gray_pixels / total_pixels
        left_white_ratio = left_white_pixels / total_pixels
        # 右半分
        right_black_pixels = np.sum(right_half <= BLACK_MAX)
        right_gray_pixels = np.sum((right_half >= GRAY_MIN) & (right_half <= GRAY_MAX))
        right_white_pixels = np.sum(right_half >= WHITE_MIN)
        right_black_ratio = right_black_pixels / total_pixels
        right_gray_ratio = right_gray_pixels / total_pixels
        right_white_ratio = right_white_pixels / total_pixels
        return (left_black_ratio, left_gray_ratio, left_white_ratio,
                right_black_ratio, right_gray_ratio, right_white_ratio)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None, None, None, None

def highlight_color_pixels(img, landmarks=None):
    """黒（黒）、灰（灰）、白（白）を強調、三角形（薄黄色）"""
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    black_mask = img <= BLACK_MAX
    img_display[black_mask] = [0, 0, 0]
    gray_mask = (img >= GRAY_MIN) & (img <= GRAY_MAX)
    img_display[gray_mask] = [150, 150, 150]
    white_mask = img >= WHITE_MIN
    img_display[white_mask] = [255, 255, 255]
    if landmarks is not None:
        for idx1, idx2 in TRIANGLE_LANDMARK_INDICES:
            x1, y1 = landmarks[idx1]
            x2, y2 = landmarks[idx2]
            px1, py1 = int(x1 + IMG_SIZE/2), int(y1 + IMG_SIZE/2)
            px2, py2 = int(x2 + IMG_SIZE/2), int(y2 + IMG_SIZE/2)
            if (0 <= px1 < IMG_SIZE and 0 <= py1 < IMG_SIZE and
                0 <= px2 < IMG_SIZE and 0 <= py2 < IMG_SIZE):
                cv2.circle(img_display, (px1, py1), 2, (200, 200, 0), -1)
                cv2.circle(img_display, (px2, py2), 2, (200, 200, 0), -1)
                cv2.line(img_display, (px1, py1), (px2, py2), (200, 200, 0), 1)
    return img_display

def check_filename_prefix(filename1, filename2, prefix_length=3):
    """ファイル名の先頭3文字が一致するかチェック"""
    try:
        prefix1 = os.path.basename(filename1)[:prefix_length]
        prefix2 = os.path.basename(filename2)[:prefix_length]
        return prefix1 == prefix2
    except Exception:
        return False

def find_similar_images(train_dir, color_ratio_threshold=COLOR_RATIO_THRESHOLD, tilt_threshold=TILT_THRESHOLD, top_n=TOP_N):
    """先頭3文字一致かつ左右半分の色割合・三角形傾き差が小さい画像ペアを特定"""
    image_files = []
    for category in ['category1', 'category2']:
        cat_dir = os.path.join(train_dir, category)
        if not os.path.exists(cat_dir):
            print(f"Directory {cat_dir} not found")
            continue
        for filename in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, filename)
            image_files.append((img_path, category))
    
    print(f"Found {len(image_files)} images")
    
    data_list = []
    for img_path, category in image_files:
        ratios = compute_color_ratios(img_path)
        landmarks = extract_landmarks(img_path)
        if all(r is not None for r in ratios) and landmarks is not None:
            data_list.append((img_path, category, *ratios, landmarks))
    
    similar_pairs = []
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            img1_path, cat1, lbr1, lgr1, lwr1, rbr1, rgr1, rwr1, lm1 = data_list[i]
            img2_path, cat2, lbr2, lgr2, lwr2, rbr2, rgr2, rwr2, lm2 = data_list[j]
            if not check_filename_prefix(img1_path, img2_path):
                continue
            # 左右半分の色割合差
            left_black_diff = abs(lbr1 - lbr2)
            left_gray_diff = abs(lgr1 - lgr2)
            left_white_diff = abs(lwr1 - lwr2)
            right_black_diff = abs(rbr1 - rbr2)
            right_gray_diff = abs(rgr1 - rgr2)
            right_white_diff = abs(rwr1 - rwr2)
            if not (left_black_diff <= color_ratio_threshold and
                    left_gray_diff <= color_ratio_threshold and
                    left_white_diff <= color_ratio_threshold and
                    right_black_diff <= color_ratio_threshold and
                    right_gray_diff <= color_ratio_threshold and
                    right_white_diff <= color_ratio_threshold):
                continue
            # 三角形3辺傾き差
            tilt_diffs = []
            for idx1, idx2 in TRIANGLE_LANDMARK_INDICES:
                t1 = compute_tilt(lm1, [idx1, idx2])
                t2 = compute_tilt(lm2, [idx1, idx2])
                if t1 == float('inf') or t2 == float('inf'):
                    diff = float('inf') if t1 != t2 else 0.0
                else:
                    diff = abs(t1 - t2)
                tilt_diffs.append(diff)
            if any(diff > tilt_threshold for diff in tilt_diffs):
                continue
            similar_pairs.append({
                'image1': img1_path,
                'image2': img2_path,
                'category1': cat1,
                'category2': cat2,
                'left_black_ratio_difference': left_black_diff,
                'left_gray_ratio_difference': left_gray_diff,
                'left_white_ratio_difference': left_white_diff,
                'right_black_ratio_difference': right_black_diff,
                'right_gray_ratio_difference': right_gray_diff,
                'right_white_ratio_difference': right_white_diff,
                'tilt_difference_eye_eye': tilt_diffs[0],
                'tilt_difference_eye_nose': tilt_diffs[1],
                'tilt_difference_nose_eye': tilt_diffs[2]
            })
    
    similar_pairs = sorted(similar_pairs, key=lambda x: (
        x['left_black_ratio_difference'],
        x['left_gray_ratio_difference'],
        x['left_white_ratio_difference'],
        x['right_black_ratio_difference'],
        x['right_gray_ratio_difference'],
        x['right_white_ratio_difference'],
        x['tilt_difference_eye_eye'],
        x['tilt_difference_eye_nose'],
        x['tilt_difference_nose_eye']
    ))
    
    if similar_pairs:
        df = pd.DataFrame(similar_pairs)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to {OUTPUT_CSV}")
    else:
        print("No similar image pairs found")
    
    print(f"\nTop {min(top_n, len(similar_pairs))} similar image pairs:")
    plt.figure(figsize=(15, 5))
    displayed = 0
    for pair in similar_pairs[:top_n]:
        img1_path = pair['image1']
        img2_path = pair['image2']
        left_black_diff = pair['left_black_ratio_difference']
        left_gray_diff = pair['left_gray_ratio_difference']
        left_white_diff = pair['left_white_ratio_difference']
        right_black_diff = pair['right_black_ratio_difference']
        right_gray_diff = pair['right_gray_ratio_difference']
        right_white_diff = pair['right_white_ratio_difference']
        tilt_diff_eye = pair['tilt_difference_eye_eye']
        tilt_diff_eye_nose = pair['tilt_difference_eye_nose']
        tilt_diff_nose_eye = pair['tilt_difference_nose_eye']
        
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                raise ValueError("Image not loaded")
            lm1 = extract_landmarks(img1_path)
            lm2 = extract_landmarks(img2_path)
            if lm1 is None or lm2 is None:
                continue
            img1_display = highlight_color_pixels(img1, lm1)
            img2_display = highlight_color_pixels(img2, lm2)
        except Exception as e:
            print(f"Error loading or processing {img1_path} or {img2_path}: {e}")
            continue
        
        plt.subplot(2, top_n, displayed + 1)
        plt.imshow(img1_display)
        plt.title(f"Image 1\n{pair['category1']}")
        plt.axis('off')
        
        plt.subplot(2, top_n, displayed + top_n + 1)
        plt.imshow(img2_display)
        plt.title(f"Image 2\nLBlack: {left_black_diff:.4f}\nLGray: {left_gray_diff:.4f}\nLWhite: {left_white_diff:.4f}\nRBlack: {right_black_diff:.4f}\nRGray: {right_gray_diff:.4f}\nRWhite: {right_white_diff:.4f}\nEye-Eye: {tilt_diff_eye:.4f}\nEye-Nose: {tilt_diff_eye_nose:.4f}\nNose-Eye: {tilt_diff_nose_eye:.4f}")
        plt.axis('off')
        
        print(f"Pair {displayed+1}:")
        print(f"  Image 1: {img1_path} ({pair['category1']})")
        print(f"  Image 2: {img2_path} ({pair['category2']})")
        print(f"  Left Black Ratio Difference: {left_black_diff:.4f}")
        print(f"  Left Gray Ratio Difference: {left_gray_diff:.4f}")
        print(f"  Left White Ratio Difference: {left_white_diff:.4f}")
        print(f"  Right Black Ratio Difference: {right_black_diff:.4f}")
        print(f"  Right Gray Ratio Difference: {right_gray_diff:.4f}")
        print(f"  Right White Ratio Difference: {right_white_diff:.4f}")
        print(f"  Eye-Eye Tilt Difference: {tilt_diff_eye:.4f}")
        print(f"  Eye-Nose Tilt Difference: {tilt_diff_eye_nose:.4f}")
        print(f"  Nose-Eye Tilt Difference: {tilt_diff_nose_eye:.4f}")
        
        displayed += 1
        if displayed >= top_n:
            break
    
    if displayed > 0:
        plt.tight_layout()
        plt.show()
    else:
        print("No pairs to display")

def main():
    try:
        find_similar_images(PREPROCESSED_TRAIN_DIR)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        face_mesh.close()

if __name__ == "__main__":
    main()