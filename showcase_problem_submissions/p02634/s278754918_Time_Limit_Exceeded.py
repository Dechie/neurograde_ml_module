import sys

from functools import lru_cache

sys.setrecursionlimit(10 ** 5)

MOD = 998244353


def solve(A, B, C, D):
    @lru_cache(maxsize=None)
    def numWays(rows, cols):
        if rows == A and cols == B:
            return 1

        assert rows >= A and cols >= B
        ways = 0
        if rows > A:
            ways += cols * numWays(rows - 1, cols) % MOD
        if cols > B:
            ways += rows * numWays(rows, cols - 1) % MOD

        if rows > A and cols > B:
            ways -= (cols - 1) * (rows - 1) * numWays(rows - 1, cols - 1) % MOD
        return ways % MOD

    return numWays(C, D) % MOD


A, B, C, D = [int(x) for x in input().split()]
print(solve(A, B, C, D))
