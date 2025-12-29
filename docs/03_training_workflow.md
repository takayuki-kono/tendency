# 学習ワークフロー・ドキュメント (Stage 3)

本ドキュメントでは、フィルタパラメータの最適化から本番学習、評価までのプロセス（Stage 3）について詳述します。

## 概要フロー
1. **最適化**: `optimize_sequential.py` (フィルタ閾値探索)
2. **学習**: `train_sequential.py` (本番モデル作成)
3. **分析**: `analyze_errors_task_a.py` (エラー分析)

---

## 1. フィルタパラメータ最適化
**スクリプト**: `optimize_sequential.py`

### 目的
「どの程度の品質の画像を捨てるべきか（あるいは残すべきか）」というトレードオフを自動的に解決します。
厳しすぎるフィルタはデータ不足を招き、緩すぎるフィルタはノイズ混入を招きます。

### アルゴリズム
Optuna等の外部ライブラリは使用せず（または補助的に使用し）、独自の**シーケンシャル最適化（順次探索）**を行います。

1. **ベースライン計測**: 全フィルタOFF状態で学習し、基準スコアを取得。
2. **パラメータ順次固定**:
   - リストアップされたパラメータ（Pitch, Symmetry, Sharpness等）を1つずつ最適化。
   - 各パラメータで `[0, 25, 50]` などの候補値を試し、最も検証スコアが良い値を採用・固定して次へ進む。
   - 必要に応じて二分探索で微調整。
3. **高速学習 (`components/train_for_filter_search.py`)**:
   - 探索時はエポック数を減らした軽量な学習ループを使用し、時間を短縮。

### キャッシュ機能
- 探索結果は `outputs/cache/parameter_cache.json` に保存。
- 途中から再開する場合、計算済みの組み合わせはスキップされる。

---

## 2. 本番学習
**スクリプト**: `train_sequential.py`

### 処理詳細
最適化されたデータセット、または手動で設定したデータセットを用いて、最終的なモデルを学習します。

- **入力**: `preprocessed_multitask/`
- **モデル**: EfficientNetV2, ResNet50 等（設定による）
- **学習戦略**:
    - **初期化**: ImageNet重みを使用。
    - **Warmup**: Head層のみを数エポック学習。
    - **Fine-tuning**: 全層（または一部）を解凍し、低学習率で再学習。
- **Data Augmentation**:
    - RandomFlip, RandomRotation, Zoom, Shift
    - **Mixup**: Alpha値を最適化。
    - **Label Smoothing**: Soft label化係数を最適化。
- **最適化パラメータ**:
    - **Model**: EfficientNetV2B0 vs EfficientNetV2S
    - **Optimizer**: Adam vs AdamW (Weight Decayの有無で自動選択)
    - **Learning Rate / Scheduler**: Base LRの探索 + Cosine Decayの適用

### 出力
- `outputs/models/best_model_task_*.h5` (またはkeras形式)
- `outputs/logs/training_log.csv` (Loss/Acc推移)

---

## 3. 評価と分析

### エラー分析 (`analyze_errors_task_a.py`)
- 検証データまたはテストデータに対する予測結果を集計。
- 混同行列 (Confusion Matrix) を出力。
- **Grad-CAM** (`visualize_gradcam.py`) を用いて、モデルが画像のどこを見て判断したかを可視化し、判断根拠を確認可能。

### バランス評価
- データ不均衡がある場合、単なるAccuracyではなく **Balanced Accuracy** を重視して評価する仕組みが組み込まれています。

---

## 補足: 署名ポリシー
コード修正時は `.agent/workflows/code-policy.md` を参照し、Author署名と仕様ドキュメントの更新を行ってください。
