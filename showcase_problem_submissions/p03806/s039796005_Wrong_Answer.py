n, ma, mb = map(int,input().split())
INF = 10000
maxa = 0
maxb = 0
maxc = 0
medi = []
for i in range(n):
    a, b, c = map(int,input().split())
    maxa = max(maxa, a)
    maxb = max(maxb, b)
    maxc = max(maxc, c)
    medi.append([a, b, c])
dp = [[[INF for k in range(maxb+1)] for j in range(maxa+1)] for i in range(n+1)]
dp[0][0][0] = 0
for i in range(n):
    for x in range(maxa+1):
        for y in range(maxb+1):
            if dp[i][x][y] == INF:
                continue
            dp[i + 1][x][y] = min(dp[i + 1][x][y], dp[i][x][y])
            a, b, c = medi[i]
            if x + a <= maxa and y + b <= maxb:
                dp[i + 1][x + a][y + b] = min(dp[i][x][y] + c, dp[i + 1][x + a][y + b])

t = 1
ok = False
while t*ma <= maxa and t*mb <= maxb:
    if dp[n][t*ma][t*mb] < INF:
        print(dp[n][t*ma][t*mb])
        ok = True
        break
    else:
        t += 1
    
if not ok:
    print(-1)