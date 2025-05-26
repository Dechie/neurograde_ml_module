#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fractions import gcd

def readln():
    _res = list(map(int,str(input()).split(' ')))
    return _res

def dp(a):
    res = [100000 for i in range(0,3000)]
    res[0] = 0
    for e in a:
        for i in range(e[0],3000):
            res[i] = min(res[i], e[1] + res[i - e[0]])
    return res

a = readln()
n,x,y = a[0],a[1],a[2]
up = []
down = []
ans = 100000
for i in range(0,n):
    a = readln()
    s = [a[0]*y-a[1]*x,a[2]]
    if s[0] > 0:
        up.append(s)
    elif s[0] < 0:
        down.append(s)
    else:
        ans = min(ans,s[1])
down = list(map(lambda x: [-x[0],x[1]],down))
fup = dp(up)
fdown = dp(down)
for i in range(1,3000):
    ans = min(ans,fup[i]+fdown[i])
if ans < 100000 : print(ans)
else : print('-1')
