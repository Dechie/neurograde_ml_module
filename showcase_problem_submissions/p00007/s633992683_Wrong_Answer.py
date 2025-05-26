r=lambda x:(x*2+1)//2
n = int(input())
ans = 100000 + (100000 * (0.05 * n))
print(int(r(ans / 10000) * 10000))