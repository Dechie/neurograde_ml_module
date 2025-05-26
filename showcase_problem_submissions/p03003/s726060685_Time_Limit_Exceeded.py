import sys

readline = sys.stdin.readline

N, M = map(int, readline().split())
S = list(map(int, readline().split()))
T = list(map(int, readline().split()))

dp = [[0] * (M + 1) for i in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, M + 1):
        dp[x][y] = dp[x][y - 1] + dp[x - 1][y] - dp[x - 1][y - 1]
        if S[x - 1] == T[y - 1]:
            dp[x][y] += dp[x - 1][y - 1] + 1
ans = (dp[N][M] + 1) % (10 ** 9 + 7)
print(ans)
