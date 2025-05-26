n = int(input())
a = [int(i) for i in input().split()]
q = int(input())
m = [int(i) for i in input().split()]

a.sort()
for ans in m:
	dp = [[0 for i in range(ans+1)] for k in range(len(a)+1)]
	dp[0][0] = 1
	for i in range(len(a)):
		for k in range(ans+1):
			if a[i] <= k:
				dp[i+1][k] = dp[i][k-a[i]] or dp[i][k]
			else:
				dp[i+1][k] = dp[i][k]
	if dp[len(a)][ans]:
		print("yes")
	else:
		print("no")
