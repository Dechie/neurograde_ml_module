n = int(input())
A = [int(x) for x in input().split()]
q = int(input())
m = [int(x) for x in input().split()]

flag = {}
for j in range(2 ** n):
    tot = 0
    for k in range(n):
        tot += ((j >> k) & 1) * A[k]
    flag[tot] = True

for i in range(q):
    if m[i] in flag:
        print("yes")
    else:
        print("no")
