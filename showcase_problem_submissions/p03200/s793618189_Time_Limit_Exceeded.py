S = list(input())

ans = 0
while True:
    tmp = 0
    for i in range(len(S)-1):
        if S[i] == "B" and S[i+1] == "W":
            S[i] = "W"
            S[i+1] = "B"
            tmp += 1
    if not tmp:
        break
    ans += tmp
        
print(ans)