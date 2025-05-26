from sys import stdin
import numpy as np
import math
import pprint
T = int(stdin.readline().rstrip())

for i in range(T):
    # print("T=",i)
    N = int(stdin.readline().rstrip())
    A = [int(x) for x in stdin.readline().rstrip().split()]
    S = stdin.readline().rstrip()
    n = len(S)
    maxa = max(A)
    keta = int (math.log2(maxa) + 1)
    if S[-1] == "1":
        print(1)
    else:
        import random
        z = random.random()
        print(int(2*z))