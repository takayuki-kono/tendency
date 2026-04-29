"""
別 venv から zibbini ConvNeXt V2 の trunk をスモークする（本番 `create_model` からは V2 を外したため直組み立て）。

実行例（リポジトリルート d:\\tendency で）:
  experimental\\convnext_v2_train_env\\venv\\Scripts\\python.exe experimental\\convnext_v2_train_env\\smoke_create_model.py
"""
import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(REPO, "components")
if COMP not in sys.path:
    sys.path.insert(0, COMP)
os.chdir(REPO)

from third_party.convnext_tf import convnext_v2 as _zcv2
from zibbini_v2_models import ZIBBINI_V2_BUILDERS

SMOKE_V2_NAME = "ConvNeXtV2Tiny"


def _div255(x):
    return tf.cast(x, tf.float32) / 255.0


if __name__ == "__main__":
    assert SMOKE_V2_NAME in ZIBBINI_V2_BUILDERS
    bname = ZIBBINI_V2_BUILDERS[SMOKE_V2_NAME]
    inp = layers.Input(shape=(224, 224, 3))
    x = layers.Lambda(_div255)(inp)
    trunk = getattr(_zcv2, bname)(input_tensor=x, include_top=False, weights=None)
    y = layers.GlobalAveragePooling2D()(trunk.output)
    m = models.Model(inp, y)
    n = m.count_params()
    print(f"smoke trunk ({SMOKE_V2_NAME} / {bname}) OK, params={n}")
