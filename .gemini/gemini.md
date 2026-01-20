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
