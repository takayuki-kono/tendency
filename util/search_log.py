
import os
target = "Google scrape finished"
try:
    with open('outputs/logs/log_part1_v2.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if target in line:
                print(line.strip())
except Exception as e:
    print(e)
