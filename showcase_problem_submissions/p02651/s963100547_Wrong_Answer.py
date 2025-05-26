t = int(input())
try_arr = []
for i in range(t):
    n = int(input())
    a = input().split()
    s = str(input())
    x = bin(0)
    try_arr.append([n,a,s,x])
for arr in try_arr:
    for i in range(arr[0]):
        if arr[2][i] == '0':
            if arr[3] != bin(0) or bin(int(arr[3], 2) ^ int(arr[1][i])) == bin(0):
                arr[3] = bin(int(arr[3], 2) ^ int(arr[1][i]))
            else:
                pass
        else:
            if arr[3] == bin(0) or bin(int(arr[3], 2) ^ int(arr[1][i])) != bin(0):
                arr[3] = bin(int(arr[3], 2) ^ int(arr[1][i]))
            elif bin(int(arr[3], 2) ^ int(arr[1][i])) != bin(0) and len(arr[1]) != i+1 and arr[2][i+1] == '0' and bin(int(arr[3], 2) ^ int(arr[1][i + 1])) == bin(0):
                arr[3] = bin(int(arr[3], 2) ^ int(arr[1][i]))
            else:
                pass
    if arr[3] == bin(0):
        print('0')
    else:
        print('1')