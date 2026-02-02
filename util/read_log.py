
import os
try:
    with open('outputs/logs/log_part1_v2.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("".join(lines[-50:]))
except Exception as e:
    print(e)
