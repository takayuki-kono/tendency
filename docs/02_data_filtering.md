# データフィルタリング・前処理ドキュメント (Stage 2)

本ドキュメントでは、収集後の画像の選別（フィルタリング）およびデータセット構成について詳述します。

## 概要フロー
1. **重複・外れ値除去**: `components/part2a`, `part2b`
2. **構造整理**: `reorganize_by_label.py`
3. **データ分割**: `create_person_split.py`
4. **詳細前処理**: `preprocess_multitask.py`

---

## 1. 初期クリーニング

### A. 類似画像除去 (`components/part2a_similarity.py`)
- **目的**: 連写やリポストによる重複画像を削除します。
- **手法**:
    - InsightFaceのFace Embedding (512次元) を抽出。
    - **DBSCANクラスタリング** (`eps`閾値設定) を行い、距離が非常に近い画像をグループ化。
    - グループ内で1枚だけ残し、他を削除。
- **CLI フラグ**:
    - `--eps <float>`: DBSCAN の `eps`（cosine 距離）。デフォルト `0.25`。小さいほど厳しく（ほぼ完全重複のみ）、大きいほど緩く（似ているだけでも同一扱い）。
    - `--min_samples <int>`: DBSCAN の `min_samples`。デフォルト `2`。
    - `--physical_delete`: 付与すると物理削除。未指定だと `deleted_duplicates/` への移動（論理削除）。
- **バッチ適用**: `run_similarity_masterdata.py` で `master_data/<category>/<person>/person_clusters/person_*` をまとめて処理可能（既定 `--eps 0.25`、物理削除は `--physical_delete` を明示）。
- **eps の調整**: 対象ディレクトリで適切な eps を探したい場合は `test_similarity_sweep.py` を使うと複数 eps 値で分岐フォルダを作って論理削除結果を並べてくれるので、エクスプローラで目視比較できる。

### B. 外れ値除去 (`components/part2b_filter.py`)
- **目的**: 検索ノイズ（同姓同名の別人、間違って検出された群衆の顔など）を除去します。
- **手法**:
    - 再びEmbeddingを用いてクラスタリング。
    - 画像枚数が多いクラスタを「本人候補」と見なし、それ以外の小規模クラスタを「外れ値（ノイズ）」として除去します。
    - デフォルトでは **上位2人物（上位2クラスタ）** を残します（同姓同名・共演者混入などで別人クラスタが大きくなるケースの救済）。
    - 残した上位クラスタは、**クラスタ順位ごとにフォルダ分け**して保存します（例: `person_clusters/person_1/`, `person_clusters/person_2/`）。

---

## 2. データセット構造化 (Stage 2a)

### ラベル整理 (`reorganize_by_label.py`)
- `master_data/{人物名}/` の構造を、学習タスクに合わせて `master_data/{ラベル}/{人物名}/` に再配置します。
- マッピング定義に基づき移動します。

### 未分類の自動ラベル付与（最新モデル推論）
`master_data/未分類/{人物フォルダ}/` に人物フォルダが溜まっている場合、**最新の学習済みモデル**で各フォルダ内の画像を推論し、
多数決でラベル（例: `adfh`）を決定して `master_data/{ラベル}/{人物フォルダ}/` に振り分けます。

- **スクリプト**: `util/reorganize_unclassified_by_model.py`
- **推論単位**: `master_data/未分類` 直下の「人物フォルダ」ごと（フォルダ内は再帰的に画像を対象）
- **ラベル決定**: 画像ごとの4タスク予測（例: `a/d/f/h`）を結合した文字列を集計し、最多ラベルを採用

### Train/Val/Test 分割 (`create_person_split.py`)
- **重要ルール**: **Person-wise Split (人物単位分割)**
- 同じ人物の画像がTrainとTestに混ざると、モデルが「顔の特徴（個人性）」を覚えてしまい、正しく汎化性能を評価できません。
- そのため、画像をランダム分割するのではなく、**「人物AはTrain、人物BはTest」**のように人物単位で振り分けます。

---

## 3. 詳細フィルタリング (Stage 2b)

### スクリプト: `preprocess_multitask.py`
最終的な学習用データセットを作成するために、画像品質に基づくフィルタリングを行います。

### 主なフィルタ項目
パラメータは `optimize_sequential.py` で最適化されることがあります。

1.  **Sharpness (鮮明度)**
    -   手法: **Laplacian Variance** (エッジの鋭さ) を計算。
    -   動作: 値が低い（ボケている）画像を除外。
2.  **Pitch (顔の縦向き)**
    -   InsightFaceのPose推定値を使用。
    -   大きく上や下を向いている顔を除外。
3.  **Symmetry (左右対称性)**
    -   左右の内眼角（目頭）と画像の中心との距離比から算出。
    -   真正面を向いているかを判定し、横顔に近い画像を除外。
4.  **Face Position (顔位置)**
    -   内眼角の座標を用いて、顔が画像の極端な端にないかを判定。
4.  **Eyebrow-Eye Distance**
    -   目と眉の距離の比率。
    -   表情のゆがみや、特徴が隠れている顔を除外。

### 出力
フィルタを通過した画像のみが `preprocessed_multitask/` フォルダに出力されます。これが学習 (`train_sequential.py`) の直接の入力となります。

### Validation クラス最小サンプル数ガード
`optimize_sequential.py` は preprocess 直後に、`preprocessed_multitask/validation/` 配下を走査して「各タスク×各クラスの画像枚数」を集計し、最小値が `MIN_VAL_PER_CLASS`（既定 20）未満の候補は学習をスキップして `score=0.0` 固定で棄却する。

- 目的: フィルタを強くしすぎると validation が激減し、small-sample ノイズで偶然の高精度（20枚に対する min class acc など）が出て採用される問題を排除する。
- タスク構造の判定: `validation/<label>/...` のフォルダ名がすべて同じ文字長 n>1 なら multitask（各文字位置を別タスクのクラスとみなす）。長さが 1 もしくは不揃いなら single-task（フォルダ名=クラス名）。
- キャッシュ挙動: 棄却された候補もキャッシュに `(0.0, total, filtered)` を書き込み、同条件の再評価を防ぐ。
