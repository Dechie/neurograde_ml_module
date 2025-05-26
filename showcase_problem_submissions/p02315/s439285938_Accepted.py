import sys
input = sys.stdin.readline

def main():
    N, W = map(int, input().split())
    vw = [list(map(int, input().split())) for i in range(N)]

    # dp[i]: 重さj以下での価値の最大値
    dp = [0] * (W+1)
    for i in range(N):
        v, w = vw[i]
        if w <= W:
            for j in range(W-w, -1, -1):
                dp[w+j] = max(dp[w+j], dp[j] + v)
    print(dp[W])

if __name__ == "__main__":
    main()
