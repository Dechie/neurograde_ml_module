import sys

read = sys.stdin.read
readline = sys.stdin.readline


class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = [[0] * (n + 1) for _ in range(26)]

    def bit_query(self, character, idx):
        if idx == 0:
            return 0
        sum_of_query = 0
        while idx > 0:
            sum_of_query += self.tree[character][idx]
            idx -= idx & (-idx)
        return sum_of_query

    def update(self, character, idx, x):
        while idx <= self.n:
            self.tree[character][idx] += x
            idx += idx & (-idx)
        return


N = int(input())
S = [''] + list(input())
Q, *query = map(str, read().split())
bit = BIT(N)
for idx, i in enumerate(S[1:], 1):
    bit.update(ord(i) - 97, idx, 1)
for a, b, c in zip(*[iter(query)] * 3):
    b = int(b)
    if a == '1':
        bit.update(ord(c) - 97, b, 1)
        bit.update(ord(S[b]) - 97, b, -1)
        S[b] = c
    else:
        c = int(c)
        print(sum(1 for i in range(26) if bit.bit_query(i, c) - bit.bit_query(i, b - 1) > 0))
