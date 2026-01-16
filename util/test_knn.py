import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

def load_insightface():
    # GPU context 0, det_size default
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

def get_embedding(app, img_path):
    try:
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: return None
        faces = app.get(img)
        if not faces: return None
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].embedding
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed_multitask", help="Root dir containing train/validation")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN")
    args = parser.parse_args()
    
    app = load_insightface()
    
    # Collect all data
    embeddings = []
    labels_dict = {'Task_A': [], 'Task_B': [], 'Task_C': [], 'Task_D': []}
    
    task_maps = {
        'Task_A': ['a', 'b', 'c'],
        'Task_B': ['d', 'e'],
        'Task_C': ['f', 'g'],
        'Task_D': ['h', 'i']
    }
    
    dirs_to_scan = [os.path.join(args.data_dir, 'train'), os.path.join(args.data_dir, 'validation')]
    
    print("Extracting features from all data...")
    count = 0
    
    for d in dirs_to_scan:
        if not os.path.exists(d): continue
        
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    fpath = os.path.join(root, file)
                    emb = get_embedding(app, fpath)
                    
                    if emb is not None:
                        embeddings.append(emb)
                        folder_name = os.path.basename(os.path.dirname(fpath))
                        
                        for task, chars in task_maps.items():
                            label = 'Unknown'
                            for c in chars:
                                if c in folder_name:
                                    label = c
                                    break
                            labels_dict[task].append(label)
                        count += 1
                        print(f"Processed {count} images...", end='\r')

    print(f"\nTotal images processed: {count}")
    if count == 0: return

    X = np.array(embeddings)
    
    print(f"\n=== k-NN Evaluation (k={args.k}) ===")
    
    for task in task_maps.keys():
        print(f"\n--- {task} ---")
        y = np.array(labels_dict[task])
        
        # Filter out 'Unknown' or incomplete labels if any
        valid_idx = y != 'Unknown'
        X_task = X[valid_idx]
        y_task = y[valid_idx]
        
        if len(np.unique(y_task)) < 2:
            print(f"Skipping {task}: Less than 2 classes found ({np.unique(y_task)})")
            continue

        # Split Train/Test (80/20) for simple evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_task, y_task, test_size=0.2, random_state=42, stratify=y_task)
        
        knn = KNeighborsClassifier(n_neighbors=args.k, metric='cosine') # Cosine similarity usually better for embeddings
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_task)))
        print(cm)
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
