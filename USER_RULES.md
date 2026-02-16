# User Global Rules (Project Level)

## Command Execution Rules
- **NEVER execute .bat files automatically.**
- Always create or update .bat files, then ask the user to execute them manually.
- This rule applies to all future interactions in this workspace.

## 【重要】グローバルルール (ユーザー指定)
以下のルールは常に優先して適用すること：
1. **開発フローの厳守 (Prompt -> Docs -> Code)**:
   - **Step 1**: ユーザーの依頼内容を `docs/prompt.md` に追記する。
   - **Step 2**: 実装前に、関連する仕様書（`pipeline_specs.md`や`docs/*.md`）を更新し、変更内容を定義する。
   - **Step 3**: コードを実装・修正する。
   - **Step 4**: `git add/commit` コマンドを提示する。
2. **ドキュメント更新の徹底**: コード修正時は必ず関連ドキュメントも更新する。
3. **Gitコマンドの共有**: 修正後は自動実行せず、git add/commit コマンドを提示する。
4. **日本語コミットメッセージ**: git commit のメッセージは必ず日本語で記述する。
5. **Author設定**: git commit 時は必ず`--author`フラグを使用する。
   - Always include the `--author` flag based on the model (Gemini/Claude).
   - This must be done proactively without the user asking.

## Prompt-Driven Development Flow (CRITICAL)
Follow this workflow for every new request or feature:
1. **Record Prompt (APPEND ONLY)**: ALWAYS **APPEND** the user's prompt verbatim to `docs/prompt.md`. Do not add headers/dates/separators, but ensure there is a single newline separator between prompts.
2. **Update Specs & Docs (MANDATORY)**: Update relevant documentation in `docs/*.md` first.
    - **ALWAYS keep docs up-to-date**: Even for small code changes, reflect them in `pipeline_specs.md` or `03_training_workflow.md`.
    - The content of `docs/prompt.md` is the source of truth and overrides existing specs.
3. **Implement**: Implement code changes ONLY based on the updated specification files.
