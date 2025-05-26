N, M = map(int, input().split())
S = [0] + list(map(int, input().split()))
T = [0] + list(map(int, input().split()))
mod = 10**9 + 7

dp = [[0] * (M+1) for i in range(N+1)]
dp_sum = [[0] * (M+1) for i in range(N+1)]

for i in range(1, N+1):
    for j in range(1, M+1):
        if S[i] == T[j]:
            dp_sum[i][j] = dp_sum[i][j-1] + dp_sum[i-1][j] + 1
        else:
            dp_sum[i][j] = dp_sum[i][j-1] + dp_sum[i-1][j] - dp_sum[i-1][j-1]

print((dp_sum[N][M]+1) % mod)
