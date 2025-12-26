import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
import argparse
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = 'best_efficient_tuned_model_EfficientNetV2B0.keras'
DATA_DIR = 'preprocessed_multitask/train' # Check training data for noise
OUTPUT_DIR = 'suspicious_labels'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.85 # Only flag if model is very sure it's wrong

# Label Definitions (Must match training)
TASK_A_LABELS = ['a', 'b', 'c']
TASK_B_LABELS = ['d', 'e']
TASK_C_LABELS = ['f', 'g']
TASK_D_LABELS = ['h', 'i']
ALL_TASK_LABELS = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]
TASK_NAMES = ["TaskA_Extension", "TaskB_Intro_Extro", "TaskC_Lateral_Vertical", "TaskD_Ear"]

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def preprocess_image(img_path):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        # EfficientNetV2 expects 0-255 inputs usually if using built-in preprocessing, 
        # but our training script used 'efficientnet_preprocess' inside the model (Lambda layer).
        # However, tf.image.resize returns float32.
        # The model input expects what?
        # In training: `x = layers.Lambda(selected_preprocess_func)(x)`
        # EfficientNetV2B0 preprocess expects 0-255.
        # So we just need to ensure it's the right shape.
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

def get_ground_truth_from_path(path):
    # Path format: .../label_name/filename.jpg
    # label_name e.g., "adfh"
    dir_name = os.path.basename(os.path.dirname(path))
    
    # Validate length
    if len(dir_name) != len(ALL_TASK_LABELS):
        return None
        
    gt_indices = []
    for i, task_labels in enumerate(ALL_TASK_LABELS):
        char = dir_name[i]
        if char in task_labels:
            gt_indices.append(task_labels.index(char))
        else:
            return None # Invalid label char
    return gt_indices, dir_name

def main():
    parser = argparse.ArgumentParser(description="Find suspicious labels in dataset")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory to scan")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    
    # Prepare output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Scanning {args.data_dir} for suspicious labels (Threshold: {args.threshold})...")

    files = []
    for root, _, filenames in os.walk(args.data_dir):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} images.")
    
    suspicious_count = 0
    
    # Process in batches? For simplicity, one by one or small batches. 
    # One by one is easier to track filenames.
    
    for file_path in tqdm(files):
        gt_result = get_ground_truth_from_path(file_path)
        if not gt_result:
            continue
        gt_indices, gt_label_str = gt_result
        
        img = preprocess_image(file_path)
        if img is None:
            continue
            
        # Predict
        preds = model.predict(img, verbose=0)
        # preds is a list of arrays [batch, num_classes] for each task
        
        # Check each task
        for i, task_pred in enumerate(preds):
            # task_pred shape (1, num_classes)
            pred_idx = np.argmax(task_pred[0])
            confidence = task_pred[0][pred_idx]
            gt_idx = gt_indices[i]
            
            if pred_idx != gt_idx and confidence >= args.threshold:
                # Found a suspicious label!
                suspicious_count += 1
                
                task_name = TASK_NAMES[i]
                gt_char = ALL_TASK_LABELS[i][gt_idx]
                pred_char = ALL_TASK_LABELS[i][pred_idx]
                
                # Create descriptive folder structure
                # suspicious_labels/TaskA/pred_b_conf0.95_gt_a/
                folder_name = f"pred_{pred_char}_conf{confidence:.2f}_gt_{gt_char}"
                save_dir = os.path.join(OUTPUT_DIR, task_name, folder_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # Copy file
                filename = os.path.basename(file_path)
                # Prefix with original full label for context
                new_filename = f"[{gt_label_str}]_{filename}"
                shutil.copy(file_path, os.path.join(save_dir, new_filename))

    print(f"\nScan complete.")
    print(f"Found {suspicious_count} suspicious instances.")
    print(f"Check the '{OUTPUT_DIR}' directory to review them.")

if __name__ == "__main__":
    main()
