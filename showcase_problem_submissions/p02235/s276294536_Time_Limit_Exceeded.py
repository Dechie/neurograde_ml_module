def lcs(maxl):
  for i in range(1, len(x)):
    for j in range(1, len(y)):
      if x[i] == y[j]:
        c[i][j] = c[i-1][j-1] + 1
      elif c[i-1][j]>c[i][j-1]:
        c[i][j] = c[i-1][j]
      else:
        c[i][j] = c[i][j-1]
      if maxl<c[i][j]:
        maxl = c[i][j]

  return maxl



q  = input()
x = [' ' for row in range(q+1)]
y = [' ' for row in range(q+1)]
c = [[0 for col in range(1001)] for row in range(1001)]
maxl = 0
for i in range(q):
  x = ' ' + raw_input()
  y = ' ' + raw_input()
  print lcs(maxl)