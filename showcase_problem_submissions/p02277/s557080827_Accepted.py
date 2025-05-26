import heapq
from collections import deque
from enum import Enum
import sys
import math
from _heapq import heappush, heappop
import copy

BIG_NUM = 2000000000
HUGE_NUM = 99999999999999999
MOD = 1000000007
EPS = 0.000000001
#sys.setrecursionlimit(3**12)


class TRAMP:
    def __init__(self,arg_mark,arg_number,arg_input_index):
        self.mark = arg_mark
        self.number = arg_number
        self.input_index = arg_input_index

N = int(input())
table = []
for input_index in range(N):
    mark,number = map(str,input().split())
    table.append(TRAMP(mark,int(number),input_index))

def Partition(left,right):
    global table
    i = left-1
    pivot = table[right].number
    for start in range(left,right):
        if table[start].number <= pivot:
            i += 1
            table[i],table[start] = table[start],table[i]
    table[i+1],table[right] = table[right],table[i+1]
    return i+1

def quickSort(left,right):
        if left < right:
            q = Partition(left,right)
            quickSort(left,q-1)
            quickSort(q+1,right)

quickSort(0, N-1)
stable_FLG = True

for i in range(1,N):
    if table[i].number == table[i-1].number and table[i].input_index < table[i-1].input_index:
        stable_FLG = False
        break

if stable_FLG:
    print("Stable")
else:
    print("Not stable")

for i in range(N):
    print("%s %d"%(table[i].mark,table[i].number))


