#! /usr/bin/env python
# -*- coding: utf-8 -*-

P = [0]*(1000100)
P[0] = 1
P[1] = 1
for i in range(2, 1000100):
    if P[i]==0:
        j = i
        for j in range(i+i, 1000100, i):
            P[j] = 1

(a, d, n) = map(int, raw_input().split())
while a!=0:
    cnt = 0
    for i in range(a, 1000100, d):
        if P[i] == 0:
            cnt += 1
        if cnt == n:
            print i
            break

    (a, d, n) = map(int, raw_input().split())