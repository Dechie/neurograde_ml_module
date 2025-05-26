n=int(input())

l=[[0 for i in range(n)] for j in range(n)]##静的確保 隣接行列
d=[0 for i in range(n)]##発見
f=[0 for i in range(n)]##完了
stack=[]##スタック
check=[]

for i in range(n):
    p=[int(x) for x in input().split()]
    
    for j in range(n):
        if j+1 in p[2:]:
            l[i][j]=1
##print("---隣接行列---")
##print(l)

time=1


while len(check)<n:
    for i in range(n):
        if i+1 not in check:
            stack+=[i+1]
            d[i]=time
            break
        else: pass

    while len(stack)!=0:
        ##print("----------while---------")
        for i in range(n):
            if l[stack[-1]-1][i]==1 and not i+1 in check and stack[-1]-1!=i:
                time+=1
                d[i]=time
                stack+=[i+1]

               ## print("stack:",stack)
               ## print("check:",check)
               ## print("d[",i,"]:",d[i])
                break
            elif i==n-1:
##最深    
                check+=[stack[-1]]
                time+=1
                f[stack.pop()-1]=time

               ## print("f[",i,"]:",f[i])
               ## print("check:",check)
    time+=1


for i in range(n):
    print(i+1," ",d[i]," ",f[i])

