L = list(map(int,input()))
N = len(L)
MOD = 10 ** 9 + 7
ans = 0
cnt = 0
for i in range(N):
    if L[i]:  
        ad = (3 ** (N - i - 1)) % MOD
        ans = ans + (2 ** cnt) * ad % MOD
        cnt += 1
print((ans + (2 ** cnt)) % MOD)