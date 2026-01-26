# AI Assistant Rules

## CRITICAL: コード修正時の必須チェックリスト
コードを修正したら、**必ず以下を全て実行**してからgitコマンドを提示すること：

1. 構文チェック（py_compile等）を実行
2. 関連する仕様書（`pipeline_specs.md`等）を更新
3. 適切なauthorでgitコマンドを提示

---

## 一般ルール
- エージェントが長時間かかるコマンド（Optunaなど）を直接バックグラウンド実行しない。バッチファイルを作成してユーザーに実行させる。
- 既存ファイルのリネームは原則行わない。ユーザーが混乱するため、名前変更が必要な場合はユーザーに確認するか、新規ファイルを作成する。

## Git Author設定
AIモデルに応じてgit commitのauthorを自動判別する：
- Claude系（Anthropic）の場合: `--author="Claude <claude@anthropic.com>"`
- Gemini系（Google）の場合: `--author="Gemini <gemini@example.com>"`

各AIは自身のモデル名を認識しているため、適切なauthorを自動選択すること。

## Git操作の禁止事項
- **Gitコマンドの自動実行は厳禁**。必ずコマンドを提示し、ユーザーにコピー＆ペーストで実行してもらうこと。
- `try_git_push.bat` などのバッチファイル形式であっても、ユーザーの許可なく自動実行してはならない。

## 【重要】グローバルルール (ユーザー指定)
以下のルールは常に優先して適用すること：
1. **ドキュメント更新の徹底**: コード修正時は必ず関連ドキュメントも更新する。
2. **Gitコマンドの共有**: 修正後は自動実行せず、git add/commit コマンドを提示する。
3. **日本語コミットメッセージ**: git commit のメッセージは必ず日本語で記述する。
4. **Author設定**: git commit 時は、操作している自身のAIモデル名（例: `--author="Gemini <gemini@google.com>"`）をAuthorに設定する。
