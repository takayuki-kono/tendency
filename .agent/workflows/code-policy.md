---
description: AIモデルによる実装の署名ルール
---

# 実装署名ルール（Author Policy）

複数のAIモデル（Gemini, Opus等）が協調して開発を行う場合、どのコードを誰が実装したかを明確にするため、以下のルールを遵守すること。

## 1. ファイルヘッダ / ドックストリング
新規ファイルの作成や、既存ファイルの主要な関数を大幅に書き換える場合、以下のような署名をコメントとして残す。

```python
"""
Module Name: example.py
Description: ...
Author: [Model Name] (e.g., Gemini 1.5 Pro)
Date: YYYY-MM-DD
"""
```

または関数単位で:

```python
def complex_algorithm():
    """
    複雑なアルゴリズムの実装
    Refactored by: Opus (YYYY-MM-DD)
    """
    pass
```

## 2. ツール実行時の説明 (Description)
`write_to_file`, `replace_file_content` などのツールを実行する際、`Description` 引数にモデル名を含めることを推奨する。
例: `Gemini: preprocess_multitask.py にキャッシュ機能を追加`

## 3. Git Author名の変更（必須）
コミット時に `--author` オプションでモデル名を明示する。

```bash
# Opusの場合
git commit --author="Claude Opus <opus@ai>" -m "コミットメッセージ"

# Geminiの場合
git commit --author="Gemini <gemini@ai>" -m "コミットメッセージ"
```

これにより `git log --author="opus"` 等でフィルタリングが可能になる。

## 4. 仕様ドキュメントの同時push（必須）
コードを修正した場合、以下を必ず同時にコミット・pushすること：

1. 修正したコードファイル
2. 修正内容を記載した仕様ドキュメント（.md形式）
   - 場所: プロジェクトルートまたは関連ディレクトリ
   - 内容: 変更の概要、理由、影響範囲など

例:
```bash
git add components/train_single_trial.py docs/training_changes.md
git commit --author="Claude Opus <opus@ai>" -m "Improve training loop with early stopping"
git push
```

---
このルールは、責任の所在を明確にし、コードの品質管理（どのモデルが得意なタスクか等）に役立てるために存在する。
