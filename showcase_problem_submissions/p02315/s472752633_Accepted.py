n, w = map(int, raw_input().split())
goods = [map(int, raw_input().split()) for i in range(n)]
dp = [0]*(w+1)
for g in goods:
    ndp = [0]*(w+1)
    for i in range(w+1):
        if i - g[1] < 0:
            ndp[i] = dp[i]
        else:
            ndp[i] = max(dp[i-g[1]] + g[0], dp[i])
    dp = ndp
print dp[w]