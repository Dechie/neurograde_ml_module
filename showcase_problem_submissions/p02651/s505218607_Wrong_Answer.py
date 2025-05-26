I=input
for i in range(int(I())):
  I()
  t=[]
  x=0
  for i,j in zip(map(int,I().split()[::-1]),I()[::-1]):
    for k in t:i^=k*(i^k<i)
    t.append(i*(j=="0"))
    x=i&(j=="1")
  print(x)