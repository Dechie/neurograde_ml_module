n,w=map(int,input().split())
w1,v1=map(int,input().split())
l1=[v1]
l2=[]
l3=[]
l4=[]
for _ in range(n-1):
    p=[int(i) for i in input().split()]
    if p[0]==w1:
        l1.append(p[1])
    if p[0]==w1+1:
        l2.append(p[1])
    if p[0]==w1+2:
        l3.append(p[1])
    if p[0]==w1+3:
        l4.append(p[1])
l1.sort()
l2.sort()
l3.sort()
l4.sort()
ans=0
for a in range(len(l1)+1):
    for b in range(len(l2)+1):
        for c in range(len(l3)+1):
            for d in range(len(l4)+1):
                if a*w1+b*(w1+1)+c*(w1+2)+d*(w1+3)<=w:
                    ans=max(ans,sum(l1[:a])+sum(l2[:b])+sum(l3[:c])+sum(l4[:d]))
print(ans)