a,b,c,d=map(int,input().split())
m=c-a +1
n=d-b+1
n_list=[[0 for i in range(m)] for j in range(n)]
n_list[0][0]=1
for i in range(1,m) :
  n_list[0][i]=n_list[0][i-1]*b

for i in range(1,n) :
  n_list[i][0]=n_list[i-1][0]*a


for i in range(1,n) :
  for j in range(1,m) :
    n_list[i][j]=n_list[i-1][j]*(a+j)+n_list[i][j-1]*(b+i)-n_list[i-1][j-1]*(a+j-1)*(b+i-1)



print(n_list[-1][-1]%998244353)
