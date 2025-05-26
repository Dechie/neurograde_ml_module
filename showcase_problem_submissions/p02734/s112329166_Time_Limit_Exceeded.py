import sys

# import bisect
# from collections import Counter, deque, defaultdict
# import copy
# from heapq import heappush, heappop, heapify
# from fractions import gcd
# import itertools
# from operator import attrgetter, itemgetter

# import math

# from numba import jit

# from scipy import
# import numpy as np
# import networkx as nx

# import matplotlib.pyplot as plt

readline = sys.stdin.readline
MOD = 10 ** 9 + 7
INF = float('INF')
sys.setrecursionlimit(10 ** 5)


def main():
    n, s = list(map(int, readline().split()))
    a = list(map(int, readline().split()))

    dp = [[0] * (s + 1) for _ in range(n)]

    for ai in range(n):
        a_cur = a[ai]
        dp[ai][0] += 1
        if a_cur <= s:
            dp[ai][a_cur] += 1
        for si in range(s + 1):
            dp[ai][si] += dp[ai - 1][si]
            if (si - a_cur) >= 0:
                dp[ai][si] += dp[ai - 1][si - a_cur]

    ans = 0
    for ai in range(n):
        ans += dp[ai][s]

    print(ans%998244353)


if __name__ == '__main__':
    main()
