# -*- coding: utf-8 -*-
import bisect
import heapq
import math
import random
import sys
from collections import Counter, defaultdict, deque
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from functools import lru_cache, reduce
from itertools import combinations, combinations_with_replacement, product, permutations
from operator import add, mul, sub, or_

sys.setrecursionlimit(100000)
input = sys.stdin.readline
INF = 2**62-1

def read_int():
    return int(input())


def read_int_n():
    return list(map(int, input().split()))


def read_float():
    return float(input())


def read_float_n():
    return list(map(float, input().split()))


def read_str():
    return input().strip()


def read_str_n():
    return list(map(str, input().split()))


def error_print(*args):
    print(*args, file=sys.stderr)



def mt(f):
    import time

    def wrap(*args, **kwargs):
        s = time.time()
        ret = f(*args, **kwargs)
        e = time.time()

        error_print(e - s, 'sec')
        return ret

    return wrap


class SegmentTree:
	def __init__(self, orig, func, unit):
		_len = len(orig)
		self.func = func
		self.size = 1 << (_len - 1).bit_length()
		self.tree = [unit] * self.size + orig + [unit] * (self.size - _len)
		self.unit = unit

		for i in range(self.size - 1, 0, -1):
			self.tree[i] = func(self.tree[i * 2], self.tree[i * 2 + 1])

	def update(self, i, v):
		i += self.size
		self.tree[i] = v
		while i:
			i //= 2
			self.tree[i] = self.func(self.tree[i * 2], self.tree[i * 2 + 1])

	def find(self, l, r):
		l += self.size
		r += self.size
		ret = self.unit
		while l < r:
			if l & 1:
				ret = self.func(ret, self.tree[l])
				l += 1
			if r & 1:
				r -= 1
				ret = self.func(ret, self.tree[r])
			l //= 2
			r //= 2
		return ret

@mt
def slv(N, S, Q):
    S = [c for c in S]
    st = SegmentTree([0] * N, or_, 0)
    for i, c in enumerate(S):
        st.update(i, 1 << (ord(c) - ord('a')))

    for q in Q:
        if q[0] == '1':
            i = int(q[1]) - 1
            c = q[2]
            st.update(i, 1 << (ord(c) - ord('a')))
        if q[0] == '2':
            i = int(q[1]) - 1
            j = int(q[2]) - 1
            ans = 0
            v = st.find(i, j+1)
            for k in range(26):
                if (1 << k) & v:
                    ans += 1
            print(ans)
    # return ans


def main():
    N = read_int()
    S = read_str()
    Q = [read_str_n() for _ in range(read_int())]
    (slv(N, S, Q))

if __name__ == '__main__':
    main()
