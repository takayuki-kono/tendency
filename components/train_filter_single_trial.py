import argparse
import os
import sys
import logging
import numpy as np
import tensorflow as tf

# Allow importing from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import BalancedSparseCategoricalAccuracy
from model_factory import create_model
from dataset_loader import get_class_names, compute_class_weights, create_dataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64 # Larger batch for speed
EPOCHS = 5      # Fixed for speed
TRAIN_DIR_NAME = 'preprocessed_single/train'
VAL_DIR_NAME = 'preprocessed_single/validation'

# Root dir
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, TRAIN_DIR_NAME)
VAL_DIR = os.path.join(PROJECT_ROOT, VAL_DIR_NAME)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B0')
    args = parser.parse_args()
    
    logger.info(f"Starting Filter Search Trial with {args.model_name}")
    
    # 1. Discover Classes
    class_names = get_class_names(TRAIN_DIR)
    num_classes = len(class_names)
    
    if num_classes == 0:
        logger.error("No classes found.")
        print("FINAL_SCORE: 0.0")
        return

    # 2. Compute Weights
    class_weights = compute_class_weights(TRAIN_DIR, class_names)
    
    # 3. Create Datasets (No augmentation for filter eval to test raw data quality)
    augment_params = {} # Empty dict = defaults (False/0.0)
    
    train_ds = create_dataset(TRAIN_DIR, class_names, IMG_SIZE, BATCH_SIZE, augment_params, shuffle=True)
    val_ds = create_dataset(VAL_DIR, class_names, IMG_SIZE, BATCH_SIZE, augment_params, shuffle=False)
    
    if train_ds is None or val_ds is None:
        print("FINAL_SCORE: 0.0")
        return

    # 4. Build Model (Fixed params for speed)
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        img_size=IMG_SIZE,
        num_dense_layers=1,
        dense_units=128,
        dropout=0.2,
        head_dropout=0.2,
        learning_rate=1e-3,
        augment_params=augment_params
    )
    
    # 5. Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=1
    )
    
    # 6. Evaluate
    logger.info("Evaluating...")
    score = model.evaluate(val_ds, verbose=0, return_dict=True)
    final_score = score.get('balanced_accuracy', 0.0)
    
    print(f"FINAL_SCORE: {final_score}")

if __name__ == "__main__":
    main()
