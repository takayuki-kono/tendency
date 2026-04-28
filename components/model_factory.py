import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetV2B0, EfficientNetV2S, ResNet50V2, ResNet101V2, Xception, DenseNet121,
    ConvNeXtSmall, MobileNetV3Large,
)
import sys
import os

# Allow importing from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import get_preprocessing_function, BalancedSparseCategoricalAccuracy, MinClassAccuracy
from zibbini_v2_models import ZIBBINI_V2_BUILDERS, normalize_zibbini_v2_model_name

def _mf_zibbini_div(x):
    return tf.cast(x, tf.float32) / 255.0

def create_model(model_name, num_classes, img_size, num_dense_layers, dense_units, dropout, head_dropout, learning_rate, augment_params):
    model_name = normalize_zibbini_v2_model_name(model_name)
    model_map = {
        'EfficientNetV2B0': EfficientNetV2B0,
        'EfficientNetV2S': EfficientNetV2S,
        'ResNet50V2': ResNet50V2,
        'ResNet101V2': ResNet101V2,
        'Xception': Xception,
        'DenseNet121': DenseNet121,
        'ConvNeXtSmall': ConvNeXtSmall,
        'MobileNetV3Large': MobileNetV3Large,
    }
    if model_name not in model_map and model_name not in ZIBBINI_V2_BUILDERS:
        model_name = 'EfficientNetV2B0'
    if model_name in ZIBBINI_V2_BUILDERS:
        BaseCnnModel = None
    else:
        BaseCnnModel = model_map[model_name]
    preprocess_func = (
        _mf_zibbini_div if model_name in ZIBBINI_V2_BUILDERS else get_preprocessing_function(model_name)
    )

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
    if model_name in ZIBBINI_V2_BUILDERS:
        _comp = os.path.dirname(os.path.abspath(__file__))
        if _comp not in sys.path:
            sys.path.insert(0, _comp)
        from third_party.convnext_tf import convnext_v2 as _zcv2
        x = layers.Lambda(_mf_zibbini_div)(x)
        builder = getattr(_zcv2, ZIBBINI_V2_BUILDERS[model_name])
        trunk = builder(input_tensor=x, include_top=False, weights=None)
        x = layers.GlobalAveragePooling2D()(trunk.output)
    else:
        x = layers.Lambda(preprocess_func)(x)
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
        BalancedSparseCategoricalAccuracy(num_classes, name='balanced_accuracy'),
        MinClassAccuracy(num_classes, name='min_class_accuracy')
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model
