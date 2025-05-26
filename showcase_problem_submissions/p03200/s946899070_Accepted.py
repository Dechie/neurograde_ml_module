S = input()
N = len(S)
res = 0
cnt = 0
for i in range(N - 1, -1, -1):
    if S[i] == 'W':
        cnt += 1
    else:
        res += cnt
print(res)
