n = input()

A = map(int, raw_input().split())

q = input()

m = map(int, raw_input().split())

def solve(i, m):
    if m == 0:
        return True
    if i >= n:
        return False
    res = solve(i + 1, m) or solve(i + 1, m - A[i])
    return res

for i in xrange(len(m)):
    if solve(0, m[i]) == True:
        print "yes"
    else:
        print "no"