N = int(input())
A = list(map(int,input().split()))
S = sum(A)
M = (S+1)//2

dp = 1
for a in A:
    dp |= (dp<<a)
ans = M
while 1:
    if dp&(1<<ans):
        print(ans)
        exit()
    ans += 1