"""
最適化・学習結果の自動記録ユーティリティ
outputs/optimization_history.jsonl に1行1JSONで追記する
"""
import json
import os
from datetime import datetime

HISTORY_FILE = os.path.join("outputs", "optimization_history.jsonl")


def log_result(script_name, result_dict):
    """
    結果を記録する。

    Args:
        script_name: 実行スクリプト名 (例: "optimize_svm_sequential")
        result_dict: 記録する結果の辞書。以下を含むことを推奨:
            - baseline_score: ベースラインスコア
            - best_score: 最終スコア
            - improvement: 改善幅
            - best_params: 最適パラメータ
            - strategy: 選択された戦略名 (任意)
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": script_name,
    }
    record.update(result_dict)

    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ResultLogger] 結果を {HISTORY_FILE} に記録しました。")
