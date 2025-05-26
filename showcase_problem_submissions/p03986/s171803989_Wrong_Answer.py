s = input()
sums = 0
sumt = 0
ans = 0
for i in s:
    if i == 'S':
        if sumt > 0:
            sumt-=1
            ans += 2
        else:
            sums += 1
    else:
        if sums > 0:
            sums -= 1
            ans += 2
        else:
            sumt += 1
print(len(s)-ans)
