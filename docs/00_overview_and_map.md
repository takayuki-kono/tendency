# プロジェクト概要とスクリプト/ドキュメント対応マップ

このドキュメントは、プロジェクト全体のスクリプト構成と、**コード変更時に更新すべきドキュメント**のマッピングを示します。

## ⚠️ 開発者向け: コード変更時のドキュメント更新ガイド

コードを変更した場合は、以下の対応表に従ってドキュメントも更新し、一緒にコミットしてください。

| 変更したスクリプト | 更新すべき主なドキュメント |
| :--- | :--- |
| **全体・共通** | |
| `pipeline_specs.md` | `docs/00_overview_and_map.md` (本書), `pipeline_specs.md` |
| **データ収集 (Stage 1)** | |
| `download_and_filter_faces.py` | `docs/01_acquisition_preprocessing.md`, `pipeline_specs.md` |
| `components/part1_setup.py` | `docs/add_google_scraping.md`, `docs/01_acquisition_preprocessing.md` |
| `components/part2a_similarity.py` | `docs/01_acquisition_preprocessing.md` |
| `components/part2b_filter.py` | `docs/01_acquisition_preprocessing.md` |
| **前処理 (Stage 2)** | |
| `reorganize_by_label.py` | `docs/01_acquisition_preprocessing.md` |
| `create_person_split.py` | `pipeline_specs.md` |
| `preprocess_multitask.py` | `docs/01_acquisition_preprocessing.md`, `pipeline_specs.md` |
| `optimize_sequential.py` | `docs/02_training_workflow.md` (作成予定) |
| **学習・評価 (Stage 3)** | |
| `train_sequential.py` | `docs/02_training_workflow.md` (作成予定), `pipeline_specs.md` |
| `components/train_for_filter_search.py` | `docs/02_training_workflow.md` |
| `analyze_errors_task_a.py` | `docs/02_training_workflow.md` |

---

## 📂 スクリプト一覧・概要

以下は、`tendency` プロジェクト内の主要スクリプトの役割と使い方のまとめです。（旧 `scripts_overview.md`）

### パイプライン（メインフロー）

| スクリプト | 役割 | 入力 | 出力 |
|-----------|------|------|------|
| `download_and_filter_faces.py` | 画像収集・初期フィルタリング | キーワードリスト | `master_data/{人物名}/` |
| `reorganize_by_label.py` | ラベル別にフォルダ再構成 | `master_data/{人物名}/` | `master_data/{ラベル}/{人物名}/` |
| `create_person_split.py` | 人物単位で train/val/test 分割 | `master_data/{ラベル}/{人物名}/` | `train/`, `validation/`, `test/` |
| `preprocess_multitask.py` | 詳細フィルタリング（sharpness等） | `train/`, `validation/`, `test/` | `preprocessed_multitask/` |
| `optimize_sequential.py` | フィルタパラメータ自動探索 | `train/` | 最適パラメータ（ログ出力） |
| `train_sequential.py` | 本番学習 | `preprocessed_multitask/` | 学習済みモデル |

### 個別スクリプト詳細（抜粋）

#### 1. download_and_filter_faces.py
- **目的**: 画像をダウンロードし、顔検出・クロップ・初期フィルタを行う。
- **処理**:
    1. `components/part1_setup.py` (Bing/Google収集 & InsightFaceクロップ)
    2. `components/part2a_similarity.py` (類似画像除去)
    3. `components/part2b_filter.py` (外れ値除去)

#### 2. components/part1_setup.py
- **目的**: 画像スクレイピングと顔検出。
- **機能追加**: Google画像検索（requests + BeautifulSoup）による高速収集機能あり（`docs/add_google_scraping.md`参照）。

#### 3. preprocess_multitask.py
- **目的**: 詳細なフィルタリング。
- **フィルタ項目**: Pitch, Symmetry, Y-Diff, Mouth Open, Eyebrow-Eye, Sharpness等。

#### 4. optimize_sequential.py
- **目的**: フィルタ閾値の自動探索（Optuna等を使用せず独自実装の場合あり）。

#### 5. train_sequential.py
- **目的**: 本番学習の実行。

### フォルダ構造

```
tendency/
├── master_data/            # ステージ1の出力
├── train/                  # 学習用分割後
├── validation/             # 検証用
├── test/                   # テスト用
├── preprocessed_multitask/ # 最終フィルタ済みデータ
├── outputs/                # ログ・モデル・キャッシュ
├── components/             # サブスクリプト群
├── docs/                   # ドキュメント集 (★ここを見る)
└── pipeline_specs.md       # 詳細仕様書
```
