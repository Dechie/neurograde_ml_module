t=int(input())
for i in range(t):
    n=int(input())
    a=list(map(int,input().split()))
    s=list(input())
    x=0
    for j in range(n):
        if s[j]=="1":
            if j!=n-1:
                if x^a[j]!=0:
                    x=x^a[j]
            else:
                if x^a[j]!=0:
                    x=10
        else:
            if x^a[j]<4:
                x=x^a[j]
    if x==0:
        print(0)
    else:
        print(1)
