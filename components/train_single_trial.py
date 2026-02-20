import argparse
import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Allow importing from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import BalancedSparseCategoricalAccuracy, MinClassAccuracy
from model_factory import create_model
from dataset_loader import get_class_names, compute_class_weights, create_dataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_DIR_NAME = 'preprocessed_single/train'
VAL_DIR_NAME = 'preprocessed_single/validation'

# Root dir (parent of components)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, TRAIN_DIR_NAME)
VAL_DIR = os.path.join(PROJECT_ROOT, VAL_DIR_NAME)

def main():
    parser = argparse.ArgumentParser()
    # Model Params
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B0')
    parser.add_argument('--num_dense_layers', type=int, default=1)
    parser.add_argument('--dense_units', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--head_dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    
    # Augmentation Params
    parser.add_argument('--rotation_range', type=float, default=0.0)
    parser.add_argument('--width_shift_range', type=float, default=0.0)
    parser.add_argument('--height_shift_range', type=float, default=0.0)
    parser.add_argument('--zoom_range', type=float, default=0.0)
    parser.add_argument('--horizontal_flip', type=str, default='False')
    parser.add_argument('--mixup_alpha', type=float, default=0.0) # Not fully implemented yet
    
    # Training Params
    parser.add_argument('--fine_tune', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.0)  # 0.0=Adam, >0=AdamW
    
    args = parser.parse_args()
    
    augment_params = {
        'rotation_range': args.rotation_range, 
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'zoom_range': args.zoom_range,
        'horizontal_flip': args.horizontal_flip.lower() == 'true',
        'label_smoothing': args.label_smoothing
    }
    
    logger.info(f"Starting Person Trial (Single Task, 24-Class) with params: {args}")
    
    # 1. Discover Classes
    class_names = get_class_names(TRAIN_DIR)
    num_classes = len(class_names)
    logger.info(f"Detected {num_classes} classes: {class_names}")
    
    if num_classes == 0:
        logger.error(f"No classes found in {TRAIN_DIR}")
        return

    # 2. Compute Class Weights
    class_weights = compute_class_weights(TRAIN_DIR, class_names)
    
    # 3. Create Datasets
    train_ds = create_dataset(TRAIN_DIR, class_names, IMG_SIZE, BATCH_SIZE, augment_params, shuffle=True)
    val_ds = create_dataset(VAL_DIR, class_names, IMG_SIZE, BATCH_SIZE, augment_params, shuffle=False)
    
    if train_ds is None or val_ds is None:
        logger.error("Failed to create datasets.")
        return

    # 4. Build Model
    model = create_model(
        args.model_name,
        num_classes,
        IMG_SIZE,
        args.num_dense_layers,
        args.dense_units,
        args.dropout,
        args.head_dropout,
        args.learning_rate,
        augment_params
    )
    
    # --- Training Setup ---
    def create_callbacks(total_epochs, initial_lr):
        def cosine_decay(epoch):
            if total_epochs == 0: return initial_lr
            return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

        return [
            EarlyStopping(monitor='val_min_class_accuracy', patience=5, restore_best_weights=True, verbose=1, mode='max'),
            LearningRateScheduler(cosine_decay, verbose=1)
        ]

    # --- Training (Single Phase) ---
    if args.fine_tune.lower() == 'true':
        logger.info(f"--- Fine-tuning Training ({args.epochs} epochs) ---")
        
        base_model_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model_layer = layer
                break
        
        if base_model_layer:
            base_model_layer.trainable = True
        
        current_lr = args.learning_rate
    else:
        logger.info(f"--- Training (Head only, {args.epochs} epochs) ---")
        current_lr = args.learning_rate

    # Always recompile
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=current_lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy', BalancedSparseCategoricalAccuracy(num_classes, name='balanced_accuracy'), MinClassAccuracy(num_classes, name='min_class_accuracy')]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=create_callbacks(args.epochs, current_lr),
        class_weight=class_weights,
        verbose=2
    )
    
    final_val_acc = max(history.history.get('val_min_class_accuracy', [0.0]))
    
    if args.fine_tune.lower() == 'true':
        model.save('best_person_model.keras')
        logger.info("Saved best fine-tuned model.")
        
    print(f"FINAL_VAL_ACCURACY: {final_val_acc}")

if __name__ == "__main__":
    main()
