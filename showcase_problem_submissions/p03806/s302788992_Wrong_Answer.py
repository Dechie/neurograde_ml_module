def examD(inf):
    N, Ma, Mb = LI()
    abc = [0,0,0]*N
    maxA = 0; maxB = 0
    for i in range(N):
        abc[i] = LI()
        if maxA<abc[i][0]:
            maxA=abc[i][0]
        if maxB<abc[i][1]:
            maxB=abc[i][1]
    dp = [[[inf]*(maxB*(N+1)+1) for _ in range(maxA*(N+1)+1)]for _ in range(N+1)]
    dp[0][0][0] = 0
    for i in range(N):
        for j in range(maxA*N+1):
            for k in range(maxB*N+1):
                dp[i+1][j+abc[i][0]][k+abc[i][1]] = min(dp[i][j+abc[i][0]][k+abc[i][1]],dp[i][j][k]+abc[i][2])
#    print(dp)
    ans = inf
    for j in range(1,maxA * N+1):
        for k in range(1,maxB * N+1):
            if j%Ma==0 and k%Mb==0:
                if j//Ma==k//Mb:
                    ans = min(ans,dp[N][j][k])
    if ans==inf:
        ans = -1
    print(ans)


import sys
import copy
from collections import Counter,defaultdict,deque
def I(): return int(sys.stdin.readline())
def LI(): return list(map(int,sys.stdin.readline().split()))
def LS(): return sys.stdin.readline().split()
def S(): return sys.stdin.readline().strip()
mod = 10**9 + 7
inf = float('inf')

examD(inf)
