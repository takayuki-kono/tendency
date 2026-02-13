import tensorflow as tf
import os
import glob
import numpy as np

def get_class_names(directory):
    """
    Scans directory for label folders.
    Structure: directory/label/person_name/image.jpg
    Returns sorted list of label names (adfh, aefh, etc.)
    """
    # Get direct subdirectories (labels)
    if not os.path.exists(directory):
        return []
    labels = [d for d in os.listdir(directory) 
              if os.path.isdir(os.path.join(directory, d))]
    return sorted(labels)

def compute_class_weights(directory, class_names):
    """
    Computes class weights for balancing.
    Returns dict {class_index: weight}
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    counts = np.zeros(len(class_names))
    
    total_images = 0
    
    for label in class_names:
        label_path = os.path.join(directory, label)
        if not os.path.isdir(label_path): continue
        
        # Count all images in the label dir AND subdirs
        idx = class_to_idx[label]
        
        # Directly under label_path
        direct_files = [f for f in os.listdir(label_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        counts[idx] += len(direct_files)
        total_images += len(direct_files)
        
        # Inside person subdirs
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)
            if os.path.isdir(item_path):
                num = len([f for f in os.listdir(item_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                counts[idx] += num
                total_images += num
            
    # Calculate weights
    num_classes = len(class_names)
    weights = {}
    for i, count in enumerate(counts):
        if count > 0:
            w = total_images / (num_classes * count)
        else:
            w = 1.0
        weights[i] = w
        
    return weights

def create_dataset(directory, class_names, img_size, batch_size, augment_params=None, shuffle=False):
    """
    Creates tf.data.Dataset from nested directory structure.
    Supports: 
    - directory/label/image.jpg
    - directory/label/person_name/image.jpg
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    file_paths = []
    labels = []
    
    print(f"DEBUG: Scanning {directory} for classes: {class_names}")
    
    for label in class_names:
        label_path = os.path.join(directory, label)
        if not os.path.isdir(label_path): 
            print(f"DEBUG: Label dir not found: {label}")
            continue
        
        # 1. Directly under label_path
        direct_files = [os.path.join(label_path, f) for f in os.listdir(label_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if direct_files:
            print(f"DEBUG: Found {len(direct_files)} images directly in {label}")
            file_paths.extend(direct_files)
            labels.extend([class_to_idx[label]] * len(direct_files))
        
        # 2. Inside subdirs (person_name)
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)
            if os.path.isdir(item_path):
                files = [os.path.join(item_path, f) for f in os.listdir(item_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if files:
                    print(f"DEBUG: Found {len(files)} images in {label}/{item}")
                    file_paths.extend(files)
                    labels.extend([class_to_idx[label]] * len(files))
            
    if not file_paths:
        print(f"Warning: No images found in {directory}")
        return None

    print(f"DEBUG: Total {len(file_paths)} images found")

    # Create Dataset
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    
    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [img_size, img_size])
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42)
    
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    
    return ds
