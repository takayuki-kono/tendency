# 別環境で `optimize_sequential` / `train_sequential` 用の TensorFlow を試す

本番（`tendency.venv_tf210_gpu`）と**分離**した venv を作り、同梱の **`components/third_party/convnext_tf`**（zibbini 由来 ConvNeXt V2）が import でき、`smoke_create_model.py` で trunk のビルドが通るか確認する。

**注意（2026-05-03）**: メイン学習の `MODEL_NAME_CANDIDATES` / `train_multitask_trial.create_model` からは ConvNeXt 系を外した。V2 の検証は本ディレクトリのスモークのみ想定。

## 1. venv 作成（例: Python 3.10）

```bat
cd d:\tendency\experimental\convnext_v2_train_env
py -3.10 -m venv venv
call venv\Scripts\activate.bat
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2. `create_model` スモーク（リポジトリの `components` を参照）

プロジェクトルートで:

```bat
cd d:\tendency
d:\tendency\experimental\convnext_v2_train_env\venv\Scripts\python.exe experimental\convnext_v2_train_env\smoke_create_model.py
```

成功例: `smoke trunk (ConvNeXtV2Tiny / convnextv2_tiny) OK, params=...`

## 3. 実学習（データが揃っている場合）

`optimize_sequential.py` / `train_sequential.py` の `PYTHON_TRAIN` / 実行 Python を **この venv の `python.exe`** に差し替えて起動する（手元の `optimize_sequential.py` 先頭の `PYTHON_TRAIN` を編集）。

- マッピングは `components/zibbini_v2_models.py` の `ZIBBINI_V2_BUILDERS`（本番パイプラインの候補には含めない）。

## 注意

- zibbini 由来の **V2 用 Imagenet 重み**はオプション。`weights=None` からの転移学習を想定。
- `components/third_party/convnext_tf` は [zibbini/convnext-v2_tensorflow](https://github.com/zibbini/convnext-v2_tensorflow) から同梱（MIT）。
