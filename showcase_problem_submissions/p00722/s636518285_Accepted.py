MAX = 1000001
isPrime = [True] * MAX
isPrime[0] = isPrime[1] = False
for i in range(2, 1000):
  if isPrime[i]:
    for j in range(i ** 2, MAX, i):
      isPrime[j] = False

def solve(a, d, n):
  if isPrime[a]:
    n -= 1
  while n:
    a += d
    if isPrime[a]:
      n -= 1
  return a

def main():
  while True:
    a, d, n = map(int, input().split())
    if a == 0:
      break
    print(solve(a, d, n))

main()
