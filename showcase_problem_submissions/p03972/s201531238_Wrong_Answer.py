
def cost(x, y, a):
    if x[0] == y[0] and abs(x[1]-y[1])==1:
        if x[1] > y[1]:
            return a[1][y[i]]
        else:
            return a[1][x[i]]
    elif x[1] == y[1] and abs(x[0]-y[0])==1:
        if x[0] > y[0]:
            return a[0][y[i]]
        else:
            return a[0][x[i]]

W = 2 
H = 2

a = [[0 for i in range(W)] for j in range(H)]

for i in range(W):
    for j in range(H):
        a[i][j] = 1#input()

a = [[3,5], [2,7]]


print(29)