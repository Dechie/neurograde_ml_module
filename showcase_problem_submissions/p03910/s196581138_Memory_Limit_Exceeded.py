from itertools import accumulate
import bisect

n = int(input())
ls = [i for i in accumulate(range(1,n+1))]
while n:
    ind = bisect.bisect_left(ls, n)
    print(ind + 1)
    n -= ind + 1