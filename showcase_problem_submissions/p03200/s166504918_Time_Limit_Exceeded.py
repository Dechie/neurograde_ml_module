S = input()

cnt = 0
idx = 0
while "BW" in S:
    if S[idx:idx+2] == "BW":
        cnt += 1
        S = S[:idx] + "WB" + S[idx+2:]
        idx = 0
    else:
      idx += 1
      
print(cnt)
