S = input()
a = []
w_lt = 0
flag = True
for i in range(len(S)):
    if S[i] == "W":
        a.append(i)
    else:
        flag = False
if not flag:
    suma = 0
    for i in range(len(a)):
        suma += a[i] - i
    print(suma)
else:
    print("0")