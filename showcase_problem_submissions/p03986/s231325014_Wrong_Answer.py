str = "TSTTSS"
res=len(str)
for i in range(len(str)-1):
       if str[i]=='S' and str[i+1]=='T':
              res = res - 2
print(res)