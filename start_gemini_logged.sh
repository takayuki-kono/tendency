#!/bin/bash

# ログファイルのパスを設定
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RAW_LOG_FILE="gemini_session_${TIMESTAMP}.raw.log"
LOG_FILE="gemini_session_${TIMESTAMP}.log"
CHAT_LOG_FILE="gemini_session_${TIMESTAMP}.chat.log"

echo "Starting Gemini CLI. Raw log: $RAW_LOG_FILE, Clean log: $LOG_FILE, Chat log: $CHAT_LOG_FILE"

# scriptコマンドで生のセッションを記録
script -q -c "gemini" "$RAW_LOG_FILE"

echo "Session ended. Cleaning log..."

# ANSIエスケープシーケンスを削除してクリーンなログを作成
# A more robust way to remove ANSI escape codes
cat "$RAW_LOG_FILE" | sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGKHF]//g" | sed -r "s/\x1b\][^\x07]*\x07//g" | sed -r "s/[\r\a]//g" > "$LOG_FILE"

# ユーザーの入力と、モデルの応答と思われる行を抽出
# Heuristic: Extract lines with user prompt or Japanese characters.
grep -E " > |[ぁ-んァ-ン一-龯]" "$LOG_FILE" | grep -v "Type your message" > "$CHAT_LOG_FILE"

echo "Clean log saved to: $LOG_FILE"
echo "Chat log saved to: $CHAT_LOG_FILE"
