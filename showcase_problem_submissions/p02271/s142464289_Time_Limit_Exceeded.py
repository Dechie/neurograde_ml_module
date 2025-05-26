n = input()
A = map(int, raw_input().split())
q = input()
M = map(int, raw_input().split())

def solve(i,m):
  if m == 0:
    return True
  if i >= n:
    return False
  res = solve(i+1,m) or solve(i+1, m - A[i])
  return res

for m in M:
    if solve(0, m):
      print 'yes'
    else:
      print 'no'