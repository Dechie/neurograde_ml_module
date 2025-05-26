n,m=map(int,input().split())
A=sorted(list(map(int,input().split())))

# M回の握手の幸福度最大化
# X以上になる握手を全て採用したときに、M通りを越える
# Xの最大値を求める

import bisect

ok=-1
ma=A[-1]
ng=(ma**2)*m+1

# x以上の握手をm通り以上作れるか
def isOk(x,m):
  cnt=0
  # 左手をループする
  for i in range(len(A)):
    left=A[i]
    minright=x-left
    # minright以上の数を数える
    cnt+=n-bisect.bisect_left(A,minright)
    if cnt>=m:
      return True
  return cnt>=m

while abs(ok-ng)>1:
  mid=abs(ok+ng)//2
  if isOk(mid,m):
    ok=mid
  else:
    ng=mid

x=ok
# x以上となる幸福度を足し合わせる
# 足し合わせた握手がMを越える場合、xぴったりを足し過ぎている

# 効率化のため累積和を作成
Asum=[0]+A
for i in range(1,len(Asum)):
  Asum[i]=Asum[i-1]+Asum[i]
  
# Aの添字i以降を全て足すときは、
# Asum[-1]-Asum[i]

ans=0
cnt=0
for i in range(len(A)):
  left=A[i]
  minright=x-left
  start=bisect.bisect_left(A,minright)
  cnt+=n-start
  ans+=left*(n-start)+(Asum[-1]-Asum[start])
  
if cnt>m:
  ans-=(cnt-m)*x
  
print(ans)
