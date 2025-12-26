# from math import gcd: 最大公約数を計算するためにgcdをインポートします。
# import sys: システム関連の機能を使うためにsysをインポートします。
from math import gcd
import sys

# Pythonの整数から文字列への変換桁数制限を解除します。
# この問題では非常に大きな数を扱う可能性があるため、この設定が必要です。
sys.set_int_max_str_digits(0)

# d（末尾iの桁数）に応じて、最適なxとiを見つけるための関数です。
def ans_d(vs, d, ud, K):
    # vsはSをKで割った余り、dはiの桁数、udはN+d、Kは割る数です。
    
    # rest = (S * 10^d) % K を計算しています。
    # vsがS%Kなので、(vs * 10^d) % K と同じ意味になります。
    rest = vs * 10**d % K

    # --- dが小さい場合 (iを全探索する) ---
    if (d <= 4):
        # ur = (10^ud) % K を計算します。これは合同方程式におけるxの係数です。
        ur = pow(10, ud, K)
        # g = gcd(ur, K) で、係数urと法Kの最大公約数を求めます。
        g = gcd(ur, K)
        
        # 最小のxとiを見つけるため、非常に大きい値で初期化しておきます。
        resx = 10**18
        resi = 0

        # 末尾の数iを、0から10^d - 1 まで全て試します。
        for i in range(10**d):
            # 合同方程式 a*x ≡ b (mod m) の b の部分を計算します。
            # rp = -(rest + i) % K
            rp = (-(rest + i)) % K

            # 合同方程式が解を持つための条件チェック。bがgcd(a,m)で割り切れない場合は解なし。
            if (rp % g != 0):
                continue
            
            # 合同方程式 (ur/g)*x ≡ (rp/g) (mod K/g) を解きます。
            # pow(a, -1, m) は、aの法mにおける逆数を計算する機能で、割り算の代わりです。
            x = pow(ur // g, -1, K // g) * (rp // g) % (K // g)

            # 見つかった解xが今までの最小値resxより小さければ、更新します。
            if (resx > x):
                resx = x
                resi = i
        # 見つかった最小のxと、そのときのiを返します。
        return resx, resi
    
    # --- dが大きい場合 (xを小さい方から試す) ---
    else:
        # ur = (10^ud) % K を計算します。
        ur = pow(10, ud, K)
        resx = 10**18
        resi = 0

        # iを全探索する代わりに、先頭の数xを小さい方から試します。
        for x in range(10**(9 - d)):
            # xを固定したときに、条件を満たすiの値(の候補)を逆算します。
            # i ≡ rp (mod K)
            rp = (-(rest + x * ur)) % K

            # 逆算したiの値(rp)が、d桁の数として妥当（rp < 10^d）であれば、
            # それが求める解です。小さいxから試しているので、最初に見つかったものが最適解となります。
            if rp < 10**d:
                return x, rp
        # ループで見つからなかった場合の戻り値です。
        return resx, resi

# 1つのテストケースを解くためのメイン関数です。
def solve():
    # KとSを読み込みます。
    K = int(input())
    S = input()
    N = len(S) # Sの長さ
    SS = int(S) # Sを整数に変換したもの

    # vs = SS % K を、桁あふれしないように計算します。
    vs = 0
    for si in S:
        vs *= 10
        vs += int(si)
        vs %= K

    # 最終的な答えの数nを記録する変数。-1は「まだ見つかっていない」印。
    ans_r = -1

    # 末尾iの桁数dを0から9まで試します。
    for d in range(10):
        # udはxにかかる10の指数 (N+d) です。
        ud = d + N
        # ans_d関数を呼び出し、このdの条件で最適なxとiを見つけてもらいます。
        x, i = ans_d(vs, d, ud, K)
        
        # 見つかったx, S, iから、元の数nを復元します。
        r = x * pow(10, ud) + SS * 10**d + i

        # 最初に見つかった答え、または、今までの答えより小さいものが見つかれば更新します。
        if (ans_r == -1):
            ans_r = r
        else:
            ans_r = min(ans_r, r)
    
    # 全てのdのパターンで最小だったものを出力します。
    print(ans_r)

# --- ここからがメインの処理 ---
# 最初にテストケースの数Tを読み込みます。
T = int(input())
# T回、solve関数を呼び出して、各テストケースを解きます。
for _ in range(T):
    solve()
