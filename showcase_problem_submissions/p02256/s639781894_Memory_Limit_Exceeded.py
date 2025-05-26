import sys
import math

x = sys.stdin.readline()

a, b = x.split(" ")

a = int(a)
b = int(b)
y = 1

if a <= b:
  a,b = b,a

c = a % b

for i in range(2, a):
  if c % i == 0 and b % i == 0:
    y *= i
    c = c / i
    b = b / i
    i -= 1

print y