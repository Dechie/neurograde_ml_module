from bisect import bisect_left
#import numpy as np
N,M = list(map(int,input().split()))
A = list(map(int,input().split()))
A.sort()

ans = [0]
for i in reversed(range(N)):
    #ans = np.r_[ans,A[i]*2 ,[A[i]] + np.array(A[i+1:n+1]),[A[i]] + np.array(A[i+1:n+1])]
    ans.append(A[i]*2)
    [ans.extend([A[i]+a,A[i]+a]) for a in A[:i]]

ans.sort()
ans = ans[-M:]

print(int(sum(ans)))