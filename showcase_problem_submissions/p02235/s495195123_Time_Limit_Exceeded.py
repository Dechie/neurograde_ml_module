for _ in range(int(input())):
    x = list(input())
    y = list(input())
    dp = [[0] * (len(y) + 1) for i in range(len(x) + 1)]
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    print(dp[-1][-1])