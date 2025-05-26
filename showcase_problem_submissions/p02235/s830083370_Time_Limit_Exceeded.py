q = int(input())

for i in range(q):
    x = list(input())
    y = list(input())
    lcs = [[0 for i in range(len(x)+1)] for j in range(len(y)+1)]
    for i in range(1,len(y) + 1):
        for j in range(1,len(x) + 1):
            if x[j-1] != y[i-1]:
                lcs[i][j] = max(lcs[i-1][j-1],lcs[i][j-1],lcs[i-1][j])
            else:
                lcs[i][j] = max(lcs[i-1][j-1] + 1,lcs[i][j-1],lcs[i-1][j])
    print(lcs[i][j])