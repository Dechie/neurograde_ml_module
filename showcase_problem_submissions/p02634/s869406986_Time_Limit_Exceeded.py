import sys
from collections import defaultdict

def solve():
    input = sys.stdin.readline
    A, B, C, D = map(int, input().split())
    mod = 998244353
    DP = defaultdict(int)
    DP[(A, B)] = 1
    for cb in range(B + 1, D + 1):
        DP[(A, cb)] = (DP[(A, cb - 1)] * A) % mod

    for ca in range(A + 1, C + 1):
        DP[(ca, B)] = (DP[(ca - 1, B)] * B) % mod
        for nb in range(B + 1, D + 1):
            DP[(ca, nb)] = (DP[(ca, nb - 1)] * ca + DP[(ca - 1, nb)] * nb - DP[(ca - 1, nb - 1)] * (ca - 1) * (nb - 1)) % mod

    print(DP[(C, D)])
    #print(DP)

    return 0

if __name__ == "__main__":
    solve()