import os
import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Initialize InsightFace
# Using standard providers. GPU recommended but CPU works for inference.
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def extract_embeddings(data_dir):
    embeddings = []
    labels = []
    paths = []
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return np.array([]), np.array([]), []

    print(f"Extracting embeddings from {data_dir}...")
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    count = 0
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            
            # Use InsightFace to detect and extract feature
            # Since images are already cropped, we assume detection will find the main face.
            # If detection fails, we might want to resize and feed directly to recognition model if possible,
            # but using app.get() is safest for alignment.
            faces = app.get(img)
            
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                if face.embedding is not None:
                    embeddings.append(face.embedding)
                    labels.append(label)
                    paths.append(fpath)
                    count += 1
            else:
                # Fallback: if detection fails (maybe too zoomed in), 
                # we could try to feed center crop directly to the recognition model,
                # but FaceAnalysis wraps it. For now, skip.
                # print(f"Skipping {fname}: No face detected")
                pass

            if count % 100 == 0:
                print(f"Processed {count} images...", end='\r')
                
    print(f"\nFinished extracting {len(embeddings)} faces from {data_dir}.")
    return np.array(embeddings), np.array(labels), paths

def main():
    parser = argparse.ArgumentParser(description="Experiment: Face Recognition Embeddings + SVM")
    parser.add_argument("--train_dir", type=str, default="train", help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="validation", help="Path to validation data")
    parser.add_argument("--use_preprocessed", action='store_true', help="Use preprocessed_multitask directory instead")
    args = parser.parse_args()

    train_dir = args.train_dir
    val_dir = args.val_dir
    
    if args.use_preprocessed:
        train_dir = os.path.join("preprocessed_multitask", "train")
        val_dir = os.path.join("preprocessed_multitask", "validation")

    print("=== Step 1: Feature Extraction ===")
    X_train, y_train_txt, _ = extract_embeddings(train_dir)
    X_val, y_val_txt, _ = extract_embeddings(val_dir)

    if len(X_train) == 0 or len(X_val) == 0:
        print("Error: No embeddings extracted. Check data paths.")
        return

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_txt)
    y_val = le.transform(y_val_txt)
    
    print(f"Classes: {le.classes_}")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Scale data (SVM likes scaled data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("\n=== Step 2: Training & Evaluation ===")
    
    classifiers = {
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, prob=True, class_weight='balanced'),
        "SVM (Linear)": SVC(kernel='linear', C=1.0, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        # print("Report:")
        # print(classification_report(y_val, y_pred, target_names=le.classes_))

    print("\n=== Experiment Complete ===")
    print("If results are good, consider using this approach instead of CNN training.")

if __name__ == "__main__":
    main()
