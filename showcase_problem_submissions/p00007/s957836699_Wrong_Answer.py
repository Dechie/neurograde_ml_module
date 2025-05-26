import math
a = 100000
n = int(input())
for _ in range(5):
    a *= 1.05
print(math.ceil(a/10000)*10000)