s = input()
if len(s)%2 == 1:
    if s[0] == s[-1]:
        print("Second")
    else:
        print("First")
else:
    if s[0] == s[-1]:
        print("First")
    else:
        print("Second")
