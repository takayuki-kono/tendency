import sys

# 標準入力から全ての行を読み取る
for i in range(1, int(input())+1, 1):
    k = int(input())
    # print(k)
    s = int(input())
    # print(s)
    for j in range(1, 1000000999999999999999999999, 1):
        # if(k*j > s):
        if str(k*j).find(str(s)) > -1:
            print(k*j)
            break
