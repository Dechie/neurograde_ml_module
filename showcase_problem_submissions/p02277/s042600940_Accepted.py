import sys
input = sys.stdin.readline

def partition(A, p, r):
    x = A[r][1]
    i = p-1
    for j in range(p, r):
        if A[j][1] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return i+1

def quicksort(A, p, r):
    if p < r:
        q = partition(A, p, r)
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)

def marge(A, left, right, mid):
    n1 = mid - left
    n2 = right - mid
    L = [[] for _ in range(n1+1)]
    R = [[] for _ in range(n2+1)]
    for i in range(n1):
        L[i] = A[left + i]
    for i in range(n2):
        R[i] = A[mid + i]
    L[n1] = ('Z', 10 ** 9 + 1)
    R[n2] = ('Z', 10 ** 9 + 1)
    i = 0
    j = 0
    for k in range(left, right):
        if L[i][1] <= R[j][1]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
    return

def margesort(A, left, right):
    if left + 1 < right:
        mid = (left + right) // 2
        margesort(A, left, mid)
        margesort(A, mid, right)
        marge(A, left, right, mid)
        return
    return


def main():
    n = int(input())
    A = [[] for _ in range(n)]
    B = [[] for _ in range(n)]
    for i in range(n):
        a, b = input().split()
        b = int(b)
        A[i] = (a, b)
        B[i] = (a, b)
    margesort(A, 0, n)
    quicksort(B, 0, n-1)
    if A == B:
        print("Stable")
    else:
        print("Not stable")
    for a in B:
        print(a[0], a[1])

if __name__ == '__main__': main()
