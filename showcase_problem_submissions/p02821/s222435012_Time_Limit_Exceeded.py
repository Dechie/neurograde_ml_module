import numpy as np
N,M = list(map(int,input().split()))
A = list(map(int,input().split()))
A.sort()

ans = []
for n,a in enumerate(A[-M:]):
  ans = np.r_[ans,[a] + np.array(A)]
  ans.sort()

print(int(sum(ans[-M:])))