t,*L=open(0)
H,W=map(int,t.split())
bef=[0]*(W+1)
a=0
for l in L:
	cur=[0]*(W+1)
	for i in range(W):
		if l[i*2]=="0":
			cur[i+1]=min(cur[i],bef[i],bef[i+1])+1
	a=max(a,max(cur))
	bef=cur
print(a**2)
