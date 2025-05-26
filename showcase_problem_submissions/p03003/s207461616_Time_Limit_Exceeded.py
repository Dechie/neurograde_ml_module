N,M = map(int,input().split())
S = list(map(int,input().split()))
T = list(map(int,input().split()))
LAST = 10**6
MOD = 10**9 + 7
S.append(LAST)
T.append(LAST)
total = [[0 for j in range(M+2)] for i in range(N+2)]
for i in range(N+1):
    for j in range(M+1):
        I,J = i+1,j+1
        if S[i] != T[j]:
            total[I][J] = (total[I][J-1] + total[I-1][J] - total[I-1][J-1]) % MOD
        else:
            total[I][J] = (total[I][J-1] + total[I-1][J] + 1) % MOD
        
print((total[N][M] + 1) % MOD)