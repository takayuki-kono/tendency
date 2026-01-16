import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from insightface.app import FaceAnalysis
from tqdm import tqdm
import pandas as pd

# 日本語フォント設定（必要に応じて）
# plt.rcParams['font.family'] = 'Meiryo'

def load_insightface():
    # GPU context 0, det_size default
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

def get_embedding(app, img_path):
    try:
        # 日本語パス対応
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: return None
        
        faces = app.get(img)
        if not faces: return None
        
        # 最大の顔を採用
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].embedding
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed_multitask/train", help="Dataset directory")
    parser.add_argument("--out_dir", type=str, default="outputs/visualization", help="Output directory for plots")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to plot to avoid overcrowding")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading InsightFace...")
    app = load_insightface()
    
    embeddings = []
    metadata = [] # List of dicts: {'filename':, 'task_a':, 'task_b':...}
    
    print(f"Scanning {args.data_dir}...")
    
    # フォルダ構造: data_dir / label_name / image.jpg
    # label_name example: "aehf" -> a, e, h, f (for Task A, B, D, C? mapping check needed)
    # create_dataset check:
    # Task A Labels: a,b,c
    # Task B Labels: d,e
    # Task C Labels: f,g
    # Task D Labels: h,i
    # Folder name maps to tasks by index. Usually: index 0->A, 1->B, 2->C, 3->D
    # BUT train_for_filter_search.py says:
    # TASK_A: 1st char? No, let's assume specific mapping or index
    
    # Actually, the folder name ITSELF is the combination label string.
    # e.g. "adfh", "behg"
    # We need to map each char to its task.
    # TaskA chars: a,b,c
    # TaskB chars: d,e
    # TaskC chars: f,g
    # TaskD chars: h,i
    
    task_maps = {
        'Task_A': ['a', 'b', 'c'],
        'Task_B': ['d', 'e'],
        'Task_C': ['f', 'g'],
        'Task_D': ['h', 'i']
    }
    
    all_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_files.append(os.path.join(root, file))
                
    if not all_files:
        print("No images found.")
        return

    # Random sample if too many
    if len(all_files) > args.max_samples:
        import random
        random.shuffle(all_files)
        all_files = all_files[:args.max_samples]
        
    print(f"Extracting features for {len(all_files)} images...")
    
    for fpath in tqdm(all_files):
        emb = get_embedding(app, fpath)
        if emb is not None:
            embeddings.append(emb)
            
            # Parse Labels from folder name
            folder_name = os.path.basename(os.path.dirname(fpath))
            
            # Determine labels by checking existence of chars
            # This is robust even if folder name order varies
            meta = {'filename': os.path.basename(fpath), 'fullpath': fpath}
            
            for task, chars in task_maps.items():
                label = 'Unknown'
                for c in chars:
                    if c in folder_name:
                        label = c
                        break
                meta[task] = label
                
            metadata.append(meta)

    if not embeddings:
        print("No valid embeddings found.")
        return
        
    X = np.array(embeddings)
    df = pd.DataFrame(metadata)
    
    print("Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(X)
    
    df['tsne_x'] = X_embedded[:, 0]
    df['tsne_y'] = X_embedded[:, 1]
    
    # Save raw data
    df.to_csv(os.path.join(args.out_dir, "tsne_data.csv"), index=False)
    
    # Plotting
    print("Generating plots...")
    tasks = list(task_maps.keys())
    
    for task in tasks:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df, x='tsne_x', y='tsne_y', hue=task, 
            palette='viridis', s=60, alpha=0.8
        )
        plt.title(f"t-SNE Visualization colored by {task}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        save_path = os.path.join(args.out_dir, f"tsne_{task}.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()
        
    print("Done! Check outputs/visualization/")

if __name__ == "__main__":
    main()
