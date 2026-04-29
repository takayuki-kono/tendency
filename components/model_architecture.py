# model_name 候補（tf.keras.applications のクラス名と一致させる）

MODEL_NAME_CANDIDATES = [
    "EfficientNetV2B0",
    "EfficientNetV2S",
    "ResNet50V2",
    "ResNet101V2",
    "MobileNetV3Large",
]

# Step1 の head LR キャリブに使う基準バックボーン（1.1 では上記候補と比較）
LRCALIB_BASE_BACKBONE = "EfficientNetV2B0"