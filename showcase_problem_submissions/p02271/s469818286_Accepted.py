from itertools import combinations
n=int(input())
a=list(map(int,input().split()))
q=int(input())
m=list(map(int,input().split()))

s=set()
for i in range(1,n+1):
    for j in combinations(a,i):
        s.add(sum(j))
for i in m:
    if i in s:
        print("yes")
    else:
        print("no")

