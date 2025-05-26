l=input()
n=len(l)
mod=1000000007
dp=[0]*n
dp[0]=2
ans=[0]*n
ans[0]=1
for i in range(1,n):
  if l[i]=='0':
    dp[i]=dp[i-1]
    ans[i]=ans[i-1]*3
  else:
    dp[i]=dp[i-1]*2
    ans[i]=(ans[i-1]*3+dp[i-1])%mod
print((ans[n-1]+dp[n-1])%mod)
