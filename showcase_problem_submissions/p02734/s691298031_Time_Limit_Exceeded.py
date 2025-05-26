n, s = map(int, input().split())
a = list(map(int, input().split()))

mod = 998244353

dp = [[0] * n for _ in range(3000 + 1)]

# init
dp[a[0]][0] = 1

for i in range(1, s + 1):
    for j in range(1, n):
        if i == a[j]:
            dp[i][j] = dp[i][j-1] + j + 1
        elif i - a[j] > 0:
            dp[i][j] = dp[i][j-1] + dp[i - a[j]][j - 1]
        else:
            dp[i][j] = dp[i][j-1]

ans = 0
for j in range(n):
    ans += dp[s][j]
    ans %= mod

print(ans % mod)
