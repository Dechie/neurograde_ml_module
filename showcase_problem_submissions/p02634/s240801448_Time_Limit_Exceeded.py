a,b,c,d=map(int, input().split())
mod=998244353
if a==c and b==d:
  print(1)
  exit()
else:
  dp=[[0]*(d+1) for _ in range(c+1)]
  for i in range(a,c+1):
    for j in range(b,d+1):
      if i==a and j==b:
        dp[i][j]=1
      else:
        dp[i][j]=dp[i][j-1]*i+dp[i-1][j]*j-dp[i-1][j-1]*(i-1)*(j-1)
        dp[i][j]%=mod
print(dp[c][d])