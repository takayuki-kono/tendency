import os
import cv2
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score
from skimage.feature import local_binary_pattern

# Configure Japanese font if needed, otherwise default
# plt.rcParams['font.family'] = 'Meiryo'

def load_insightface():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

def extract_color_histogram(img, bins=32):
    # HSV Histogram (Color)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp(img, points=24, radius=3):
    # Texture (LBP)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_geometry(face):
    # 2D Landmark features (Geometry)
    if face.kps is None: return np.zeros(10)
    
    # Simple geometry: relative distances
    # 0:LeftEye, 1:RightEye, 2:Nose, 3:LeftMouth, 4:RightMouth
    kp = face.kps
    
    # Normalize by eye distance
    eye_dist = np.linalg.norm(kp[0] - kp[1]) + 1e-6
    
    feats = []
    # Aspect ratios / geometries
    feats.append(np.linalg.norm(kp[0] - kp[3]) / eye_dist) # Left Eye-Mouth
    feats.append(np.linalg.norm(kp[1] - kp[4]) / eye_dist) # Right Eye-Mouth
    feats.append(np.linalg.norm(kp[2] - kp[0]) / eye_dist) # Nose-LeftEye
    feats.append(np.linalg.norm(kp[2] - kp[1]) / eye_dist) # Nose-RightEye
    feats.append(np.linalg.norm(kp[2] - (kp[3]+kp[4])/2) / eye_dist) # Nose-MouthCenter
    
    return np.array(feats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="master_data", help="Source directory (master_data)")
    parser.add_argument("--out_dir", type=str, default="outputs/clustering_evaluation_master", help="Output directory")
    parser.add_argument("--sample_limit", type=int, default=1000, help="Max images to process to save time")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading InsightFace...")
    app = load_insightface()
    
    # Data Collection from master_data (Recursive)
    # master_data/Label/Person/Image.jpg
    
    image_paths = []
    if os.path.exists(args.data_dir):
        for root, dirs, files in os.walk(args.data_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(root, f))
    else:
        print(f"Directory not found: {args.data_dir}")
        return

    if args.sample_limit > 0 and len(image_paths) > args.sample_limit:
        import random
        random.shuffle(image_paths)
        image_paths = image_paths[:args.sample_limit]
        
    print(f"Processing {len(image_paths)} images from {args.data_dir}...")
    
    features = {
        'insightface': [],
        'color': [],
        'texture': [],
        'geometry': []
    }
    
    # Metadata for labels
    labels = {
        'Task_A': [], 'Task_B': [], 'Task_C': [], 'Task_D': [],
        'filename': []
    }
    
    task_maps = {
        'Task_A': ['a', 'b', 'c'],
        'Task_B': ['d', 'e'],
        'Task_C': ['f', 'g'],
        'Task_D': ['h', 'i']
    }
    
    valid_indices = []

    for i, path in enumerate(tqdm(image_paths)):
        try:
            # Load Image
            img_stream = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
            if img is None: continue
            
            # InsightFace & Geometry
            faces = app.get(img)
            if not faces: continue
            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            
            # Extract All Features
            emb = face.embedding
            geom = extract_geometry(face)
            col = extract_color_histogram(img)
            tex = extract_lbp(img)
            
            features['insightface'].append(emb)
            features['geometry'].append(geom)
            features['color'].append(col)
            features['texture'].append(tex)
            
            # Labels
            # path: master_data/Label/Person/Image.jpg
            # folder_name (Person): os.path.basename(os.path.dirname(path))
            # label_folder (Label): os.path.basename(os.path.dirname(os.path.dirname(path)))
            
            parent_dir = os.path.dirname(path)
            grandparent_dir = os.path.dirname(parent_dir)
            label_folder = os.path.basename(grandparent_dir)
            
            labels['filename'].append(os.path.basename(path))
            
            for task, chars in task_maps.items():
                lbl = 'unknown'
                for c in chars:
                    if c in label_folder:  # Check the LABEL FOLDER (bdgh etc), not Person folder
                        lbl = c
                        break
                labels[task].append(lbl)
                
            valid_indices.append(i)
            
        except Exception as e:
            # print(f"Error {path}: {e}")
            pass

    # Convert to numpy arrays
    X_dict = {k: np.array(v) for k, v in features.items()}
    
    # Create combined features (e.g. InsightFace + Geometry)
    # Standardize first
    scalers = {}
    X_scaled = {}
    for k, v in X_dict.items():
        if len(v) == 0: continue
        scalers[k] = StandardScaler()
        X_scaled[k] = scalers[k].fit_transform(v)
        
    # Add Combined
    if 'insightface' in X_scaled and 'geometry' in X_scaled:
        X_scaled['deep_geo'] = np.hstack([X_scaled['insightface'], X_scaled['geometry']])
        
    print("\nEvaluating Clustering Methods...")
    
    results = []
    
    # Methods to test
    # We use KMeans with k=number of classes in task, essentially checking convertibility
    
    for task_name, task_labels in labels.items():
        if task_name == 'filename': continue
        
        # Filter valid targets
        y_true = np.array(task_labels)
        mask = y_true != 'unknown'
        if np.sum(mask) < 10: continue
        
        y_target = y_true[mask]
        n_clusters = len(np.unique(y_target))
        if n_clusters < 2: continue
        
        print(f"--- Evaluaring for {task_name} ({n_clusters} classes) ---")
        
        for feat_name, X in X_scaled.items():
            X_curr = X[mask]
            
            # Clustering Algorithms
            algos = {
                'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
                # 'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters), # Slow for large data
            }
            
            for algo_name, model in algos.items():
                try:
                    y_pred = model.fit_predict(X_curr)
                    
                    # Metrics
                    # V-Measure: 0.0-1.0 (1.0 = perfect match with ground truth)
                    v_score = v_measure_score(y_target, y_pred)
                    
                    # ARI: -1.0-1.0 (1.0 = perfect match, 0.0 = random)
                    ari = adjusted_rand_score(y_target, y_pred)
                    
                    # Silhouette: -1.0-1.0 (Cluster separation quality, independent of label)
                    sil = silhouette_score(X_curr, y_pred)
                    
                    results.append({
                        'Task': task_name,
                        'Feature': feat_name,
                        'Algorithm': algo_name,
                        'V-Measure': v_score,
                        'ARI': ari,
                        'Silhouette': sil
                    })
                    
                    # Visualization (t-SNE) for top features only?
                    # Let's save plot for every feature for now
                    
                    # t-SNE projection
                    if len(X_curr) > 1000: # Limit tsne samples
                        X_tsne = X_curr[:1000]
                        y_tsne = y_target[:1000]
                        pred_tsne = y_pred[:1000]
                    else:
                        X_tsne = X_curr
                        y_tsne = y_target
                        pred_tsne = y_pred
                        
                    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                    X_emb = tsne.fit_transform(X_tsne)
                    
                    # Plot: Ground Truth Colors
                    plt.figure(figsize=(12, 5))
                    
                    plt.subplot(1, 2, 1)
                    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=y_tsne, palette='viridis', s=50)
                    plt.title(f"Ground Truth: {task_name}\n({feat_name})")
                    
                    plt.subplot(1, 2, 2)
                    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=pred_tsne, palette='Set2', s=50, legend=False)
                    plt.title(f"Clustering: {algo_name}\n(V-Measure: {v_score:.3f})")
                    
                    plt.tight_layout()
                    safe_feat = feat_name.replace(' ', '_')
                    plt.savefig(os.path.join(args.out_dir, f"{task_name}_{safe_feat}_{algo_name}.png"))
                    plt.close()
                    
                except Exception as e:
                    print(f"Error in {feat_name}-{algo_name}: {e}")
                    
    # Save Results
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by=['Task', 'V-Measure'], ascending=[True, False])
        csv_path = os.path.join(args.out_dir, "clustering_evaluation_report.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"\nReport saved to: {csv_path}")
        print("\nTop 5 Feature Combinations per Task:")
        for task in labels.keys():
            if task == 'filename': continue
            rows = df_res[df_res['Task'] == task].head(3)
            if not rows.empty:
                print(f"\n--- {task} ---")
                print(rows[['Feature', 'Algorithm', 'V-Measure', 'Silhouette']].to_string(index=False))
    else:
        print("No valid results computed.")

if __name__ == "__main__":
    main()
