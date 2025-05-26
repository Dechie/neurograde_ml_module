md=10**9+7
n,m=map(int,input().split())
s=list(input().split())
t=list(input().split())
dp=[[0]*(m+3) for _ in range(n+3)]
dp[0][0]=1
for i,sk in enumerate(s):
    for j,tk in enumerate(t):
        if sk==tk:
            dp[i+1][j+1]=sum(sum(dpi[:j+1]) for dpi in dp[:i+1])
print(sum(sum(dpi) for dpi in dp)%md)