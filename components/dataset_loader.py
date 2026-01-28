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
        
        # Count all images in all person subdirs
        for person_name in os.listdir(label_path):
            person_path = os.path.join(label_path, person_name)
            if not os.path.isdir(person_path): continue
            
            num = len([f for f in os.listdir(person_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            
            idx = class_to_idx[label]
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
    Class = label folder name (adfh, aefh, etc.)
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
        
        print(f"DEBUG: Checking label dir: {label}")
        
        # Collect all images from all person subdirs
        for person_name in os.listdir(label_path):
            person_path = os.path.join(label_path, person_name)
            if not os.path.isdir(person_path): continue
            
            files = [os.path.join(person_path, f) for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if files:
                print(f"DEBUG: Found {len(files)} images in {label}/{person_name}")
            
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
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42)
    
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    
    return ds
