def partition(p, r):
    i = p
    for j in range(p, r):
        if A[r][1] >= A[j][1]:
            A[i], A[j] = A[j], A[i]
            i += 1
    A[r], A[i] = A[i], A[r]
    return i

def quick_sort(p, r):
    if p < r:
        q = partition(p, r)
        quick_sort(p, q-1)
        quick_sort(q+1, r)

def merge(left, mid, right):
    L = merge_list[left:mid]
    R = merge_list[mid:right]
    L.append(["", INFTY])
    R.append(["", INFTY])
    i, j = 0, 0
    for k in range(left, right):
        if L[i][1] <= R[j][1]:
            merge_list[k] = L[i]
            i += 1
        else:
            merge_list[k] = R[j]
            j += 1

def merge_sort(left, right):
    if left+1 < right:
        mid = int( (left+right)/2 )
        merge_sort(left, mid)
        merge_sort(mid, right)
        merge(left, mid, right)

INFTY = 1000000001

N = int(input())
A = []
for _ in range(N):
    suit, num = input().split()
    num = int(num)
    A.append([suit, num])
merge_list = A[:]

quick_sort(0, N-1)
merge_sort(0, N)

if A == merge_list:
    print("Stable")
else:
    print("Not stable")

for card in A:
    print("%s %d" % (card[0], card[1]))

