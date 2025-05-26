num = int(input())
x = 0
for i in range(num):
  ran = int(input())
  su = map(int,input().split())
  man = list(map(int,(input())))
  for count,k in enumerate(su):
    ans = bin(x ^ k)
    #もしlistの中身
    if man[count] == 0 and x == 0 and int(ans,0) != 0:
      pass
    elif man[count] == 1 and int(ans,0) == 0:
      pass
    else:
      x = int(ans,0)
  if x == 0:
    print(0)
  else:
    print(1)
  x = 0