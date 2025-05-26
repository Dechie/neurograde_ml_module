

N,M=[int(c) for c in input().split()]
A=[int(c) for c in input().split()]
B=sorted([x+y for x in A for y in A],reverse=True)
ans=0
for i in range(M):
  ans+=B[i]

print(ans)



