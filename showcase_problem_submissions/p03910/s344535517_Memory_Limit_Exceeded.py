import itertools
#import bisect

N = int(input())
L = [i for i in range((N+1)//2+2)]

acu = list(itertools.accumulate(L))
#print(acu) 
#index = bisect.bisect_left(acu,N)
#print(index)

for i in range(len(acu)):
    if N == acu[i]:
        index = i
        Max = acu[i-1]
        break
    elif N < acu[i]:
        index = i
        Max = acu[i]
        break
Left = Max - N    
#if acu[index] == N:
#    Max = acu[index-1]
#else:
#    Max = acu[index]
#Left = Max - N

for i in range(1,index+1):
    if i == Left:
        continue
    print(i)