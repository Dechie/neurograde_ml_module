def lcs(s, t):
  slen = len(s)
  tlen = len(t)
  dp = [0] * (slen + 1)
  for i in range(slen + 1):
    dp[i] = [0] * (tlen + 1)

  for i in range(1, slen + 1):
    for j in range(1, tlen + 1):
      if s[i-1] == t[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
      else:
        dp[i][j] = max(dp[i][j-1], dp[i-1][j])

  return dp[slen][tlen]

n = int(raw_input())

for i in range(n):
  x = raw_input()
  y = raw_input()
  print(lcs(x, y))