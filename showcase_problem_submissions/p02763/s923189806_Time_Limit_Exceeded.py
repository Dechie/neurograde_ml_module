class BinaryIndexedTree:
    def __init__(self,n,default = 0):
        self.s = [default]*(n+1)
        self.n = n
    
    def add(self,val,idx):
        while idx < self.n+1:
            self.s[idx] = self.s[idx] + val
            idx += idx&(-idx)
        return
    
    def get(self,idx):
        res = 0
        while idx > 0:
            res = res + self.s[idx]
            idx -= idx&(-idx)
        return res

import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

n = int(input())
S = [ord(s) - ord('a') for s in input()[:-1]]

BIT = [BinaryIndexedTree(n) for i in range(26)]

for i,s in enumerate(S):
    BIT[s].add(1,i+1)

Q = int(input())
ans = []
for _ in range(Q):  
    flag,x,y = input().split()

    if flag == "1":
        i = int(x)
        c = ord(y) - ord('a')
        BIT[S[i-1]].add(-1,i)
        BIT[c].add(1,i)
        S[i-1] = c
    else:
        l = int(x)
        r = int(y)
        res = 0
        for i in range(26):
            # print([BIT[i].get(r),BIT[i].get(l)])
            res += (BIT[i].get(r) > BIT[i].get(l-1))
        ans.append(str(res))
print("\n".join(ans))
