import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2S, ResNet50V2, Xception, DenseNet121
import sys
import os

# Allow importing from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import get_preprocessing_function, BalancedSparseCategoricalAccuracy

def create_model(model_name, num_classes, img_size, num_dense_layers, dense_units, dropout, head_dropout, learning_rate, augment_params):
    model_map = {
        'EfficientNetV2B0': EfficientNetV2B0,
        'EfficientNetV2S': EfficientNetV2S,
        'ResNet50V2': ResNet50V2,
        'Xception': Xception,
        'DenseNet121': DenseNet121
    }
    BaseCnnModel = model_map.get(model_name, EfficientNetV2B0)
    preprocess_func = get_preprocessing_function(model_name)

    # Data Augmentation
    aug_layers = []
    if augment_params.get('horizontal_flip'):
        aug_layers.append(layers.RandomFlip("horizontal"))
        
    aug_layers.extend([
        layers.RandomRotation(augment_params.get('rotation_range', 0.0)),
        layers.RandomZoom(augment_params.get('zoom_range', 0.0)),
        layers.RandomTranslation(
            height_factor=augment_params.get('height_shift_range', 0.0), 
            width_factor=augment_params.get('width_shift_range', 0.0)
        ),
    ])
    data_augmentation = tf.keras.Sequential(aug_layers)

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = layers.Lambda(preprocess_func)(x)

    # Base Model (Transfer Learning)
    base_model = BaseCnnModel(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
    base_model.trainable = False 
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(head_dropout)(x)

    # Dense Layers
    for _ in range(num_dense_layers):
        x = layers.Dense(dense_units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x)

    # Output Layer (Single Task - 24 classes)
    # Mixed Precision stability: dtype='float32'
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    
    # Loss (SparseCategoricalCrossentropy doesn't support label_smoothing in TF 2.10)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Metrics
    metrics = [
        'accuracy',
        BalancedSparseCategoricalAccuracy(num_classes, name='balanced_accuracy')
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model
