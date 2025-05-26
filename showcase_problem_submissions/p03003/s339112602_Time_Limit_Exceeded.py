source = raw_input()
input = source.split(" ")
n, m = int(input[0]), int(input[1])
s = [0] + [int(i) for i in raw_input().split(" ")]
t = [0] + [int(i) for i in raw_input().split(" ")]
MOD = 10**9 + 7

dp = [[0] * (m + 1) for _ in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, m + 1):
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1]
        if s[i] == t[j]:
            dp[i][j] += dp[i - 1][j - 1] + 1
        dp[i][j] %= MOD

print((dp[n][m] + 1) % MOD)