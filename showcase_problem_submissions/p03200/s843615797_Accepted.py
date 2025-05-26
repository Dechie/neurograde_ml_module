import sys
readline = sys.stdin.buffer.readline
readlines = sys.stdin.buffer.readlines
read = sys.stdin.buffer.read
sys.setrecursionlimit(10 ** 7)
INF = float('inf')

S = list(input())

ans = 0
w_count = 0
for i, s in enumerate(S):
    if s == "W":
        ans += i - w_count
        w_count += 1

print(ans)