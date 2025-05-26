import bisect
import itertools

N,M=[int(i) for i in input().split(" ")]

A=[int(i) for i in input().split(" ")]
A=sorted(A,reverse=True)
A=A[:M]
scores=[]
for pair in itertools.product(A, repeat=2):
    score=pair[0]+pair[1]
    index = bisect.bisect_left(scores, score)
    scores.insert(index,score)


print(sum(scores[-M:]))
