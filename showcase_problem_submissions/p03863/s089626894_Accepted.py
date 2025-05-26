s = list(input())
print("First" if (len(s) - len(set([s[0], s[-1]]))) % 2 == 1 else "Second")