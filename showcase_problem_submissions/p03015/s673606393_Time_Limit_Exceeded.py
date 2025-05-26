L=input()
n=len(L)
mod=10**9+7
table=[None]*n
table[0]=1
for i in range(1,n):
    table[i]=3*table[i-1]%mod
ans=0
k=1
for i in range(n):
    if L[i]=="1":
        ans=(ans+k*table[n-i-1])%mod
        k*=2
ans+=k    
print(ans%mod)