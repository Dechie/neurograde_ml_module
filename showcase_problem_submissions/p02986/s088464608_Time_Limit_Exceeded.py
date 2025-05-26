import sys
sys.setrecursionlimit(100000000)
input = sys.stdin.readline
from math import log
from collections import deque

def solve():
    N, Q = map(int, input().split())
    es = [[] for i in range(N)]
    for i in range(N-1):
        u, v, c, d = map(int, input().split())
        u -= 1
        v -= 1
        es[u].append([v, c, d])
        es[v].append([u, c, d])
    LOG_N = int(log(N, 2)) + 1
    parent = [[-1] * N for i in range(LOG_N)]
    level = [None] * N
    level[0] = 0
    dist = [None] * N
    dist[0] = 0
    dq = deque()
    dq.append(0)
    while dq:
        v = dq.popleft()
        p = parent[0][v]
        lv = level[v]
        dv = dist[v]
        for u, c, d in es[v]:
            if u != p:
                parent[0][u] = v
                level[u] = lv + 1
                dist[u] = dv + d
                dq.append(u)
    for k in range(LOG_N-1):
        parentk = parent[k]
        for v in range(N):
            if parentk[v] < 0:
                parent[k+1][v] = -1
            else:
                parent[k+1][v] = parentk[parentk[v]]

    def lca(u, v):
        if level[u] > level[v]:
            t, u = u, v
            v = t
        for k in range(LOG_N)[::-1]:
            if (level[v] - level[u]) >> k & 1:
                v = parent[k][v]
            if level[v] == level[u]:
                break
        if u == v:
            return u
        for k in range(LOG_N)[::-1]:
            if parent[k][u] != parent[k][v]:
                u, v = parent[k][u], parent[k][v]
        return parent[0][u]

    qs = [[] for i in range(N)]
    query = [None] * Q
    for i in range(Q):
        x, y, u, v = map(int, input().split())
        u -= 1
        v -= 1
        qs[u].append([i, x, y, 1])
        qs[v].append([i, x, y, 1])
        a = lca(u, v)
        qs[a].append([i, x, y, -2])
        query[i] = dist[v] + dist[u] - 2 * dist[a]
    ds = [0] * N
    cnt = [0] * N

    def EulerTour(v, p):
        for id, col, y, f in qs[v]:
            query[id] += f * (y * cnt[col] - ds[col])
        for u, c, d in es[v]:
            if u != p:
                ds[c] += d
                cnt[c] += 1
                EulerTour(u, v)
                ds[c] -= d
                cnt[c] -= 1

    EulerTour(0, -1)
    for res in query:
        print(res)

if __name__ == '__main__':
    solve()
