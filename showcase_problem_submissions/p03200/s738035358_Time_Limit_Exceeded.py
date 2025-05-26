a = list(input())
ans = 0
while True:
    j = 0
    for i in range(len(a) - 1):
        if a[i] == "B" and a[i+1] == "W":
            a[i] = "W"
            a[i+1] = "B"
            j = 1
            ans += 1
            break
    if j == 0:
        break
print(ans)