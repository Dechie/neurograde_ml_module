s = list(input())
count = 0
iter_flag = True
while iter_flag:
    ope_flag = False
    for i in range(len(s)-1):
        if s[i]+s[i+1] == "BW":
            s[i] = "W"
            s[i+1] = "B"
            count += 1
            ope_flag = True
        if i == len(s)-2 and ope_flag == False:
            iter_flag = False
print(count)