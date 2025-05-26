s = str(input())
a = len(s)%2==0
b = s[0]==s[-1]

if a ^ b:
    print('Second')
else:
    print('First')