# プロジェクト概要とスクリプト/ドキュメント対応マップ

このドキュメントは、プロジェクト全体のスクリプト構成と、**コード変更時に更新すべきドキュメント**のマッピングを示します。

## ⚠️ 開発者向け: コード変更時のドキュメント更新ガイド

コードを変更した場合は、以下の対応表に従ってドキュメントも更新し、一緒にコミットしてください。

| 変更したスクリプト | 更新すべき主なドキュメント |
| :--- | :--- |
| **全体・共通** | |
| `pipeline_specs.md` | `docs/00_overview_and_map.md` (本書), `pipeline_specs.md` |
| **データ収集 (Stage 1)** | |
| `download_and_filter_faces.py` | `docs/01_data_acquisition.md`, `pipeline_specs.md` |
| `components/part1_setup.py` | `docs/add_google_scraping.md`, `docs/01_data_acquisition.md` |
| **フィルタリング・前処理 (Stage 2)** | |
| `components/part2a_similarity.py` | `docs/02_data_filtering.md` |
| `components/part2b_filter.py` | `docs/02_data_filtering.md` |
| `reorganize_by_label.py` | `docs/02_data_filtering.md` |
| `create_person_split.py` | `docs/02_data_filtering.md`, `pipeline_specs.md` |
| `preprocess_multitask.py` | `docs/02_data_filtering.md`, `pipeline_specs.md` |
| **学習・評価 (Stage 3)** | |
| `optimize_sequential.py` | `docs/03_training_workflow.md` |
| `train_sequential.py` | `docs/03_training_workflow.md`, `pipeline_specs.md` |
| `components/train_for_filter_search.py` | `docs/03_training_workflow.md` |
| `analyze_errors_task_a.py` | `docs/03_training_workflow.md` |

---

## 📂 スクリプト一覧・概要

`master_data/` への収集から `preprocessed_multitask/` への前処理、そして学習への流れを制御するスクリプト群です。

### パイプライン（メインフロー）

| スクリプト | 役割 | 入力 | 出力 | 対応Doc |
|-----------|------|------|------|:---:|
| `download_and_filter_faces.py` | 画像収集・初期フィルタ | キーワード | `master_data/` | `01` |
| `reorganize_by_label.py` | フォルダ再構成 | `master_data/` | `master_data/` | `02` |
| `create_person_split.py` | データ分割 (Train/Val/Test) | `master_data/` | `train/`等 | `02` |
| `preprocess_multitask.py` | 詳細フィルタ・正規化 | `train/`等 | `preprocessed/` | `02` |
| `optimize_sequential.py` | パラメータ最適化 | `train/` | パラメータ | `03` |
| `train_sequential.py` | 本番学習 | `preprocessed/` | モデル | `03` |

### フォルダ構成

```
tendency/
├── master_data/            # Stage 1 出力
├── train/                  # Stage 2a 出力
├── validation/
├── test/
├── preprocessed_multitask/ # Stage 2b 出力 (学習入力)
├── outputs/                # ログ・モデル
├── components/             # 補助スクリプト
└── docs/                   # ドキュメント集
    ├── 00_overview_and_map.md      # 全体マップ
    ├── 01_data_acquisition.md      # データ収集
    ├── 02_data_filtering.md        # フィルタリング・前処理
    ├── 03_training_workflow.md     # 学習ワークフロー
    └── 04_development_rules.md     # 開発ルール
```
