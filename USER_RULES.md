# User Global Rules (Project Level)

## Command Execution Rules
- **NEVER execute .bat files automatically.**
- Always create or update .bat files, then ask the user to execute them manually.
- This rule applies to all future interactions in this workspace.

## 【重要】グローバルルール (ユーザー指定)
以下のルールは常に優先して適用すること：
1. **開発フローの厳守 (Prompt -> Docs -> Code)**:
   - **Step 1 (Critical)**: ユーザーの依頼内容を `docs/prompt.md` に**追記(APPEND)**する。**重要: 最終行を上書きしないよう、必ずファイル末尾を確認してから追記すること。**
   - **Step 2**: 実装前に、関連する仕様書（`pipeline_specs.md`や`docs/*.md`）を更新し、変更内容を定義する。
   - **Step 3**: コードを実装・修正する。
   - **Step 4**: `git add/commit` コマンドを提示する。
2. **ドキュメント更新の徹底**: コード修正時は必ず関連ドキュメント (.md) も更新すること。**Prompt -> Docs -> Code の順序を厳守。**
3. **Gitコマンドの共有**: 修正後は自動実行せず、git add/commit コマンドを提示する。
4. **日本語コミットメッセージ (絶対厳守)**: git commit のメッセージは**必ず日本語**で記述すること。**英語のメッセージは禁止**。コミット前に必ず確認せよ。
5. **Author設定**: git commit 時は必ず`--author`フラグを使用する。
   - Always include the `--author` flag based on the model (Gemini/Claude).
   - This must be done proactively without the user asking.

6. **日本語出力の遵守**: 思考プロセス、計画、説明など、ユーザーへの応答は**可能な限り日本語**で行うこと。
   - コード内のコメントやリテラル文字列はこの限りではないが、説明文は日本語とする。

## Prompt-Driven Development Flow (CRITICAL - ABSOLUTE PRIORITY)
Follow this workflow for every new request or feature:
1. **Record Prompt (IMMEDIATELY & APPEND ONLY)**:
   - User requests MUST be appended to `docs/prompt.md` **before** any other action.
   - **Format**: `- **YYYY-MM-DD**: "User's exact prompt content"`
   - **CRITICAL**: Do NOT overwrite the last line. Always append a NEW line. Check the file end before writing.
2. **Update Specs & Docs (MANDATORY)**:
   - Update relevant documentation in `docs/*.md` (e.g., `03_training_workflow.md`) **before** coding.
   - Reflect ALL changes, including small logic tweaks or parameter updates.
   - The content of `docs/prompt.md` acts as the source of truth.
3. **Implement Code**:
   - Implement code changes based on the updated specs.
4. **Git Commit (JAPANESE ONLY)**:
   - Always verify that the commit message is in **Japanese**.
   - Use `--author` flag.
