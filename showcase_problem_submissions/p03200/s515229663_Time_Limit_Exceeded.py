S = list(input())

cnt = 0
t = -1
while cnt != t:
    t = cnt
    for i in range(len(S)-1):
        if S[i] == 'B' and S[i+1] == 'W':
            S[i] = 'W'
            S[i+1] = 'B'
            cnt += 1

print(cnt)