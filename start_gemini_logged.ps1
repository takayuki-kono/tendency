# =============================================================================
# Gemini CLI 自動ログ記録起動スクリプト
# =============================================================================
#
# 使い方:
# 1. 下の「▼▼▼ 編集が必要な箇所 ▼▼▼」セクションにあるコマンドを、
#    お使いのGemini CLIを起動する実際のコマンドに書き換えてください。
# 2. このファイルを右クリックし、「PowerShellで実行」を選択します。
#
# =============================================================================

# ログファイルのパスを設定（C:\tendencyに固定名で作成）
$logPath = "C:\tendency\gemini_session.log"

Write-Host "Attempting to start log recording to: C:\tendency\gemini_session.log"

# ログ記録を開始
try { Start-Transcript -Path "C:\tendency\gemini_session.log" -Append -Force; Write-Host "Start-Transcript succeeded." } catch { Write-Host "Error starting transcript: $($_.Exception.Message)"; exit 1 }

Write-Host "Log recording started. Log file: C:\tendency\gemini_session.log"
Write-Host "Launching Gemini CLI..."


Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -NoExit -Command 'cd C:\tendency; gemini; Read-Host ''Press Enter to close this Gemini CLI window.'''" -Wait


Write-Host "Reached end of Gemini CLI execution block."

Read-Host "Gemini CLI session ended. Press Enter to close this window."

# スクリプトが終了すると、ログ記録も自動的に停止します。
