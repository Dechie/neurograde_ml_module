import re

def gcd(x, y):
    if x == y:
        return x
    if x < y and y % x == 0:
        return x
    if x > y and x % y == 0:
        return y


    l = list(range(1, int(min(x, y) / 2) + 1))
    l.reverse()
    r = 1
    #for i in range(1, int(min(x, y) / 2) + 1):
    for i in l:
        if x % i == 0 and y % i == 0:
            return i
    return 1
 
x, y = map(int, re.split(' ', raw_input()))
print(gcd(x, y))