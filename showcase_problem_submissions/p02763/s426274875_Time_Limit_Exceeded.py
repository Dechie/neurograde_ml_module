n = int(input())
s = list(input())
q = int(input())
for i in range(q):
    a, b, c =map(str, input().split())
    b = int(b)
    a = int(a)
    if(a==2):
        c = int(c)
        p = s [b-1:c]
        #print(p)
        print(len(set(p)))
    if(a==1):
        s[b-1] = c
