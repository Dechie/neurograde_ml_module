def main():
    inf = 100 * 40 + 1
    N, Ma, Mb = map(int, input().split())

    dpa = [inf] * (400 + 1)
    dpa[0] = 0
    dpb = [inf] * (400 + 1)
    dpb[0] = 0
    # dp[x]:=物質をxg用意する最小費用

    for _ in range(N):
        a, b, price = map(int, input().split())
        for g in range(400, -1, -1):
            dpa[g] = min(dpa[g], dpa[g - a] + price)
            dpb[g] = min(dpb[g], dpb[g - b] + price)

    ans = inf
    k = 1
    while max(Ma * k, Mb * k) <= 400:
        cost = dpa[Ma * k] + dpb[Mb * k]
        if cost < ans:
            ans = cost
        k += 1

    if ans == inf:
        print(-1)
    else:
        print(ans)


if __name__ == '__main__':
    main()

# import sys
#
# sys.setrecursionlimit(10 ** 7)
#
# input = sys.stdin.readline
# rstrip()
# int(input())
# map(int, input().split())
