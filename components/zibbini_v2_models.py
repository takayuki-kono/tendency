# zibbini 由来 ConvNeXt V2（`third_party/convnext_tf`）。値は `convnext_v2` モジュール上の builder 関数名。
# 同梱に `convnextv2_small` は無いため、探索用の 2 枠目（ConvNeXtV2Small）は `convnextv2_nano`（参数量: nano < tiny）
ZIBBINI_V2_BUILDERS = {
    "ConvNeXtV2Tiny": "convnextv2_tiny",
    "ConvNeXtV2Small": "convnextv2_nano",
}

# 旧 `MODEL_NAME_CANDIDATES` / JSON の `ConvNeXtV2Atto` → 現行 Tiny
def normalize_zibbini_v2_model_name(model_name):
    if model_name == "ConvNeXtV2Atto":
        return "ConvNeXtV2Tiny"
    return model_name
