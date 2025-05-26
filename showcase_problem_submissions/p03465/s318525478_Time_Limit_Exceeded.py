#!/usr/bin/env python

from collections import deque
import itertools as it
import sys
import math

sys.setrecursionlimit(10000000)

N = input()
A = map(int, raw_input().split())

m_p2 = {}
def p2(num):
    if num in m_p2:
        return m_p2[num]
    if num == 0:
        m_p2[num] = 1
        return 1
    m_p2[num] = p2(num - 1) * 2
    return m_p2[num]

ans = 1
for num in A:
    ans |= ans * p2(num)

S = sum(A)
mid = S / 2 + S % 2

for i in range(mid, S + 1):
    if ans & p2(i):
        print i
        break
