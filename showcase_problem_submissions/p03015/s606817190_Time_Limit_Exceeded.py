from collections import defaultdict,deque
import sys,heapq,bisect,math,itertools,string,queue,datetime
sys.setrecursionlimit(10**8)
INF = float('inf')
mod = 10**9 + 7
eps = 10**-7
def inp(): return int(sys.stdin.readline())
def inpl(): return list(map(int, sys.stdin.readline().split()))
def inpl_str(): return list(sys.stdin.readline().split())

S = input()

def solve(S):
    L = len(S)
    if L == 0 or int(S) == 0:
        return 1

    tmp = 0
    tmpS = ''
    for i in range(L):
        if S[i] == '1':
            tmpS = S[i+1:]
            tmp =  L - 1 - i
            break


    return (2*solve(tmpS) + solve_bin(tmp))%mod

def solve_bin(b):
    return pow(3,b,mod)

print(solve(S)%mod)
