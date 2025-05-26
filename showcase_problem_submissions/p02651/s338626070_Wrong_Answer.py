t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    s = input()
    p1 = []
    p0 = []
    for i, ss in enumerate(s):
        p1.append(a[i]) if ss == "1" else p0.append(a[i])
    res1 = 0
    for p11 in p1:
        res1 |= p11
    res0 = 0
    for p00 in p0:
        res0 |= p00
    bin1 = str(bin(res1))
    bin0 = str(bin(res0))
    if len(bin1) != len(bin0):
        print(1)
    else:
        print(0)
