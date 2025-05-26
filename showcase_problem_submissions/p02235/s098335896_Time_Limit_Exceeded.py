import sys
sys.setrecursionlimit(10000)
def lcs(X, Y):
    i = len(X)
    j = len(Y)
    if not i or not j:
        return 0
    if memo[i][j] >= 0:
        return memo[i][j]
    if X[-1] == Y[-1]:
        memo[i][j] = 1 + lcs(X[:-1], Y[:-1])
        return memo[i][j]
    else:
        memo[i][j] = max(lcs(X[:-1], Y), lcs(X, Y[:-1]))
        return memo[i][j]

q = int(input())
for i in range(q):
    X = input()
    Y = input()
    memo = [[-1 for j in range(len(Y) + 1)] for i in range(len(X) + 1)]
    print(lcs(X, Y))