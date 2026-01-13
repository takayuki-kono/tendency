---
description: コード修正後のgitコマンド提示ルール
---

# コード修正時のルール

## 絶対に守ること

1. **gitコマンドは自動実行しない**
   - git status, git add, git commit, git push 等、すべてのgitコマンドは自動実行禁止
   - 代わりに、ユーザーがコピペで実行できるコマンドを提示する

2. **コード修正後の手順**
   - 構文チェック（py_compile等）を実行して確認
   - 関連する仕様書(.md)があれば更新
   - 以下の形式でgitコマンドを提示：

```cmd
cd /d d:\tendency
git add <変更したファイル>
git commit -m "日本語のコミットメッセージ" --author="Gemini <gemini@example.com>"
git push origin master
```

3. **コミットメッセージは日本語で書く**

4. **すべての応答は日本語で行う**
