# User Global Rules (Project Level)

<RULE>
## 【最重要・AI強制プロトコル】開発フローの絶対厳守 (Prompt -> Docs -> Code)
AIアシスタントは、いかなる作業を開始する前にも、自身の思考プロセス（Thought block）内で「私は Prompt -> Docs -> Code の順序を必ず守ります」と宣言・確認し、以下の1〜4のステップを**順番通り確実に**実行しなければならない。これを忘れたり、順序をスキップすることは固く禁じます。

### 離陸前チェック（各ターン、最初のファイル編集ツール呼び出しの前に完了）
- [ ] 1. `docs/prompt.md` に本ターンのユーザー発話を末尾追記（StrReplace等のファイル編集ツール限定）
- [ ] 2. 挙動/仕様変更があれば、関連 `docs/*.md` を更新
- [ ] 3. コード修正
- [ ] 4. 修正したPyファイルを `python -m py_compile` で検査（エラー0）
- [ ] 5. `git add` / `git commit --author="<Model> <mail>" -m "<日本語>"` を提示

1. **Prompt (CRITICAL & APPEND ONLY)**
   - ユーザーから受けた新しい依頼内容は、いかなる解析やコード修正の作業の**一番最初**に `docs/prompt.md` へ追記すること。
   - **追記手段の固定**: `docs/prompt.md` の追記は**必ずファイル編集ツール（StrReplace 等）で末尾に追加**すること。PowerShell の `Add-Content` / `echo` / `Out-File` 等、シェル経由の追記は日本語＋引用符のエスケープ事故が発生するため**禁止**。
   - **フォーマット固定（2行テンプレ）**:
     ```md
     - **YYYY-MM-DD**: "ユーザー発話そのまま（要約/改変禁止）"
       - 対応: 変更点（ファイル名/関数/要点）
     ```
   - 最終行を上書きしないよう、末尾に新しい行として追記(APPEND)すること。
2. **Docs (MANDATORY)**
   - 実際のコード変更を行う前に、関連する仕様書（`docs/*.md`、`pipeline_specs.md`、`03_training_workflow.md`等）に変更内容・設計・仕様を反映すること。
   - 事前に何を変更するかドキュメント化してから実装に入る。
3. **Code**
   - ドキュメントで定義した仕様に基づいて実装や改修を行う。
   - 修正後は、必ず構文チェック（py_compile等）を実行してエラーがないことを確認すること。
4. **Git共有 (JAPANESE ONLY & AUTHOR FLAG)**
   - 修正後は `git add/commit` コマンドをユーザーに**提示**すること（AI側で git commit 等を自動で実行してはいけない）。
   - **不要なGitコマンドは提示しない**: ユーザーから明示的に要求されない限り、`git status` / `git diff` の提示は行わない（`git add` / `git commit` / `git push` に必要な最小限の提示に留める）。
   - **コミットメッセージは必ず日本語**で記述すること。英語のメッセージは禁止。
   - コマンドには必ず**当該チャットで稼働しているエージェント／モデル表示名**に基づく `--author` を付与すること。セッションが切り替わると別モデルになるため、**固定の第三者名にハードコードしない**（例: Cursor の Composer 利用時は `--author="Composer <composer@cursor.agent>"` のように、実際の表示名に合わせる。Gemini 専用チャットなら `Gemini <...>` 等）。ダミーの汎用名（`Your Name` 等）は使わないこと。
   - **シェル互換（厳守, cmd.exe 最優先）**: ユーザー環境は **Windows の cmd.exe（コマンドプロンプト）を最優先**（PowerShell は二の次）。コマンド提示時のコードブロックは必ず ``` ```bat ``` ``` を使い、以下は**禁止**:
       - bash の here-doc（`"$(cat <<'EOF' ... EOF)"`）
       - bash の行継続バックスラッシュ `\`
       - PowerShell のバックティック `` ` `` 継行／PowerShell 固有コマンド
       - `;` / `&&` / `&` でのコマンド連結（1行1コマンドで改行区切り）
   - **多行コミットメッセージ**: 必ず `-m "..."` を**複数回指定**する形式で提示する（1個目=件名、2個目以降=本文段落）。メッセージ内に改行は入れない。
     ```bat
     git commit --author="Composer <composer@cursor.agent>" -m "件名" -m "- 変更点1" -m "- 変更点2"
     ```
     （`--author` の名前は**そのターンのエージェント名**に置き換え。詳細は `.cursor/rules/git-workflow.mdc`。）
</RULE>

---

## Command Execution Rules
- **NEVER execute .bat files automatically.**
- Always create or update .bat files, then ask the user to execute them manually.
- This rule applies to all future interactions in this workspace.

## その他の遵守事項
- **日本語出力の遵守**: 思考プロセス、計画、説明など、ユーザーへの応答は**可能な限り日本語**で行うこと。（コード内のコメント等はこの限りではないが、説明文は日本語とする）
- **既存ファイル名維持**: 既存ファイルのリネームは原則行わない。必要な場合はユーザーに確認すること。
