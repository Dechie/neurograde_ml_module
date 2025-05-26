s = input()
n = len(s)
print("Second" if (n&1)==(s[0]==s[-1]) else "First")