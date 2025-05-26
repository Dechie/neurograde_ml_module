def merge(A, l, m, r):
  L = A[l:m]
  L.append((1e10, None))
  R = A[m:r]
  R.append((1e10, None))
  i, j = 0, 0
  for k in range(l, r):
    if L[i][0] <= R[j][0]:
      A[k] = L[i]
      i += 1
    else:
      A[k] = R[j]
      j += 1

def msort(A, l, r):
  if r-l > 1:
    m = (l+r)//2
    msort(A, l, m)
    msort(A, m, r)
    merge(A, l, m, r)

def partition(A, l, r):
    v = A[r-1]
    i = l - 1
    for j in range(l, r-1):
        if A[j][0] <= v[0]:
            i += 1
            t = A[i]
            A[i] = A[j]
            A[j] = t
    q = i+1
    A[r-1] = A[q]
    A[q] = v
    return q

def qsort(A, l, r):
    if r-l > 1:
        q = partition(A, l, r)
        qsort(A, l, q)
        qsort(A, q+1, r)

N = int(input())
A, B = [], []
for _ in range(N):
    c, n = input().split()
    n = int(n)
    A.append((n, c))
    B.append((n, c))
qsort(A, 0, N)
msort(B, 0, N)
print('Stable' if A == B else 'Not stable')
for n, c in A:
    print(c, n)
