import os
import cv2
import numpy as np
import argparse
import shutil
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from tqdm import tqdm

def load_insightface():
    # Use GPU if available
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

def calculate_aspect_ratio(app, img_path):
    try:
        # Handle Japanese paths
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: return None, None
        
        faces = app.get(img)
        if not faces: return None, None
        
        # Select largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        
        # Use 2D landmarks (106 points if available, else 68 or 5)
        # Using landmark coordinates is more accurate than bbox for face shape
        # because bbox includes margin and sometimes hair context.
        
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            pts = face.landmark_2d_106
        elif hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            pts = face.landmark_3d_68[:, :2] # Drop Z
        elif hasattr(face, 'kps') and face.kps is not None:
            pts = face.kps
        else:
            # Fallback to bbox if absolutely no landmarks (unlikely with FaceAnalysis)
            w = face.bbox[2] - face.bbox[0]
            h = face.bbox[3] - face.bbox[1]
            return h / (w + 1e-6), img
            
        # Calculate bounding box of LANDMARKS only (pure face area)
        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        ratio = height / (width + 1e-6)
        
        return ratio, img
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

def save_image(img, out_path):
    # Ensure dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Save with Japanese path support
    ext = os.path.splitext(out_path)[1]
    result, enc_img = cv2.imencode(ext, img)
    if result:
        with open(out_path, mode='w+b') as f:
            enc_img.tofile(f)

def main():
    parser = argparse.ArgumentParser(description="Classify faces by Aspect Ratio (Long vs Wide)")
    parser.add_argument("--src_dir", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--out_dir", type=str, default="outputs/face_shape_classification", help="Output directory")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for Aspect Ratio (Height/Width). If None, uses Median.")
    parser.add_argument("--action", type=str, choices=['analyze', 'copy', 'move'], default='analyze', help="Action to perform")
    args = parser.parse_args()
    
    app = load_insightface()
    
    image_files = []
    for root, dirs, files in os.walk(args.src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
                
    if not image_files:
        print("No images found.")
        return

    print(f"Analyzing {len(image_files)} images...")
    
    ratios = []
    results = [] # list of (path, ratio)
    
    for fpath in tqdm(image_files):
        ratio, _ = calculate_aspect_ratio(app, fpath)
        if ratio is not None:
            ratios.append(ratio)
            results.append((fpath, ratio))
            
    if not ratios:
        print("Could not extract any face info.")
        return
        
    ratios = np.array(ratios)
    
    # Statistics
    mean_val = np.mean(ratios)
    median_val = np.median(ratios)
    min_val = np.min(ratios)
    max_val = np.max(ratios)
    
    print("\n" + "="*40)
    print(f"Aspect Ratio Statistics (Height / Width)")
    print(f"Count: {len(ratios)}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print("="*40)
    
    # Determine Threshold
    threshold = args.threshold if args.threshold is not None else median_val
    print(f"\nUsing Threshold: {threshold:.4f}")
    print(f"Long Face (Vertical) > {threshold}")
    print(f"Wide Face (Horizontal) <= {threshold}")
    
    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label=f'Threshold: {threshold:.2f}')
    plt.title('Face Aspect Ratio Distribution')
    plt.xlabel('Ratio (Height / Width)')
    plt.ylabel('Count')
    plt.legend()
    
    hist_path = os.path.join(args.out_dir, "aspect_ratio_histogram.png")
    os.makedirs(args.out_dir, exist_ok=True)
    plt.savefig(hist_path)
    print(f"Histogram saved to: {hist_path}")
    
    # Action
    if args.action in ['copy', 'move']:
        long_dir = os.path.join(args.out_dir, "long")
        wide_dir = os.path.join(args.out_dir, "wide")
        os.makedirs(long_dir, exist_ok=True)
        os.makedirs(wide_dir, exist_ok=True)
        
        print(f"\nPerforming '{args.action}' operation...")
        
        for fpath, ratio in tqdm(results):
            fname = os.path.basename(fpath)
            # Avoid name collision if flat structure
            # fname = f"{ratio:.2f}_{fname}" 
            
            if ratio > threshold:
                dest_path = os.path.join(long_dir, fname)
            else:
                dest_path = os.path.join(wide_dir, fname)
            
            if args.action == 'copy':
                # Use shutil for copy but handle japanese paths? shutil works usually.
                # If not, use read/write
                try:
                    shutil.copy2(fpath, dest_path)
                except:
                    # Fallback copy
                    with open(fpath, 'rb') as fsrc:
                        with open(dest_path, 'wb') as fdst:
                            shutil.copyfileobj(fsrc, fdst)
            elif args.action == 'move':
                try:
                    shutil.move(fpath, dest_path)
                except:
                    shutil.copy2(fpath, dest_path)
                    os.remove(fpath)
                    
        print(f"Done. Check {long_dir} and {wide_dir}")
    else:
        print("\nAction was 'analyze'. No files moved. Use --action copy or --action move to organize files.")

if __name__ == "__main__":
    main()
