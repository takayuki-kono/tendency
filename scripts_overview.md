# スクリプト一覧・概要

本ドキュメントは、`tendency` プロジェクト内の主要スクリプトの役割と使い方を簡潔にまとめたものです。

---

## パイプライン（メインフロー）

| スクリプト | 役割 | 入力 | 出力 |
|-----------|------|------|------|
| `download_and_filter_faces.py` | 画像収集・初期フィルタリング | キーワードリスト | `master_data/{人物名}/` |
| `reorganize_by_label.py` | ラベル別にフォルダ再構成 | `master_data/{人物名}/` | `master_data/{ラベル}/{人物名}/` |
| `create_person_split.py` | 人物単位で train/val/test 分割 | `master_data/{ラベル}/{人物名}/` | `train/`, `validation/`, `test/` |
| `preprocess_multitask.py` | 詳細フィルタリング（sharpness等） | `train/`, `validation/`, `test/` | `preprocessed_multitask/` |
| `optimize_sequential.py` | フィルタパラメータ自動探索 | `train/` | 最適パラメータ（ログ出力） |
| `train_sequential.py` | 本番学習 | `preprocessed_multitask/` | 学習済みモデル |

---

## 個別スクリプト詳細

### 1. download_and_filter_faces.py
**目的**: 画像をダウンロードし、顔検出・クロップ・初期フィルタを行う。
**処理**:
1. `components/part1_setup.py` を呼び出し（ダウンロード＆クロップ）
2. `components/part2a_similarity.py` を呼び出し（類似画像除去）
3. `components/part2b_filter.py` を呼び出し（外れ値除去・整理）

**設定**: `KEYWORDS` リストにダウンロード対象の人物名を記載。

---

### 2. components/part1_setup.py
**目的**: 画像ダウンロード、顔検出、回転補正、クロップ。
**処理**:
1. BingImageCrawler で画像をダウンロード
2. InsightFace で顔検出・ランドマーク取得
3. 顔を正立に回転補正
4. 眉〜顎でクロップし、224x224 にリサイズ
5. `{output_dir}/rotated/` に保存

---

### 3. components/part2a_similarity.py
**目的**: 類似画像（ほぼ同じ画像）を検出・削除する。
**処理**: 特徴量ベクトル（InsightFace embedding）を計算し、DBSCANで重複をクラスタリング・削除。
**設定**:
- `PHYSICAL_DELETE = True`: 物理削除
- `PHYSICAL_DELETE = False`: `deleted/` フォルダへ移動

---

### 4. components/part2b_filter.py
**目的**: 他人の顔（外れ値）を除去し、フォルダ整理を行う。
**処理**:
1. embeddingを用いたクラスタリングで最大クラスタ以外を削除
2. 残った画像を `rotated/` から親ディレクトリへ移動
3. 空になった `rotated/` フォルダを削除
**設定**:
- `PHYSICAL_DELETE = True`: 物理削除
- `PHYSICAL_DELETE = False`: `deleted/` フォルダへ移動

---

### 5. reorganize_by_label.py
**目的**: 人物名ベースの構造をラベルベースに変換する。
**処理**:
- `master_data/{人物名}/` → `master_data/{ラベル}/{人物名}/`
**設定**: `LABEL_MAPPING` 辞書に人物→ラベルのマッピングを記載。

---

### 6. create_person_split.py
**目的**: 人物単位で train/validation/test に分割する。
**ルール**:
- 各ラベル(クラス)内に3人以上 → 1人目train, 2人目val, 3人目test
- 2人 → 1人目train, 2人目val
- 1人 → trainのみ
**入力**: `master_data/{ラベル}/{人物名}/`
**出力**: `train/{ラベル}/{人物名}/`, `validation/...`, `test/...`

---

### 7. preprocess_multitask.py
**目的**: 詳細なフィルタリングを行い、学習用データセットを生成する。
**フィルタ項目**:
- Pitch（顔の上下向き）
- Symmetry（左右対称性）
- Y-Diff（傾き）
- Mouth Open（口の開き）
- Eyebrow-Eye Distance（眉と目の距離） ← 個人別統計で閾値計算
- **Sharpness（画像の鮮明度）** ← Laplacian Varianceでぼやけ検出

**引数例**:
```bash
python preprocess_multitask.py --sharpness_percentile_low 5 --eyebrow_eye_percentile_high 10
```

**デフォルト設定**:
- `SHARPNESS_PERCENTILE_LOW = 5` (下位5%のぼやけ画像を除外)

---

### 8. optimize_sequential.py
**目的**: 各フィルタパラメータを自動探索し、最適な設定を見つける。
**処理**:
1. ベースライン (全パラメータ=0) を最初に評価してキャッシュ
2. 探索点 [0, 5, 50] を評価
3. 上位2点の間で二分探索
4. 各パラメータを順番に最適化

**探索順序**:
1. Pitch
2. Symmetry
3. Y-Diff
4. Mouth Open
5. Eb-Eye High
6. Eb-Eye Low
7. **Sharpness Low**
8. Face Position (True/False)

**スコア抽出**: `FINAL_SCORE: X.XXX` を `train_for_filter_search.py` の出力から取得

**出力**: ログに最適パラメータを表示

---

### 9. train_sequential.py
**目的**: 本番学習を実行する。
**処理**: `preprocessed_multitask/` を入力として学習を行い、モデルを保存。

---

### 10. components/train_for_filter_search.py
**目的**: フィルタ探索用の軽量学習（エポック数少なめ）。
**用途**: `optimize_sequential.py` から呼び出される。
**出力**: `FINAL_SCORE: {score}` を標準出力に表示。

---

## ユーティリティスクリプト

| スクリプト | 役割 |
|-----------|------|
| `reorganize_master_data.py` | フォルダ構造を整理（古い構造から新構造へ移行） |
| `filter_similar_preprocessed.py` | 前処理後データの類似画像除去 |
| `analyze_errors_task_a.py` | 学習結果のエラー分析 |
| `check_landmarks.py` | ランドマーク可視化確認用 |
| `count_data.py` | データ数カウント |
| `organize_outputs.py` | 出力フォルダを整理 |
| `organize_scripts.py` | スクリプトを整理・分類 |

---

## フォルダ構造

```
tendency/
├── master_data/            # ステージ1の出力（ラベル別に整理後）
│   └── {ラベル}/
│       └── {人物名}/
│           └── (画像ファイル)
├── train/                  # 学習用（create_person_split.py の出力）
├── validation/             # 検証用
├── test/                   # テスト用
├── preprocessed_multitask/ # preprocess_multitask.py の出力
├── outputs/                # 全出力ファイル
│   ├── logs/               # 各スクリプトのログファイル
│   ├── cache/              # 最適化キャッシュ (JSON)
│   └── models/             # 学習済みモデル (.h5, .keras)
├── components/             # サブスクリプト群
├── pipeline_specs.md       # パイプライン仕様書
└── scripts_overview.md     # 本ドキュメント
```

