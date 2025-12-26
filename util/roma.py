import random

def generate_simple_katakana(length=5):
    # """
    # 基本的なカタカナ46文字からランダムな文字列を生成する
    # """
    # ア〜ンまでの基本的なカタカナ
    katakana_chars ="アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"

    # 指定された長さのランダムな文字列を生成
    random_word = "".join(random.choices(katakana_chars, k=length))

    return random_word

# --- 実行 ---
print("--- パターン1：基本的なカタカナ ---")
for _ in range(100):  # 5個生成して表示
    print(generate_simple_katakana(5))