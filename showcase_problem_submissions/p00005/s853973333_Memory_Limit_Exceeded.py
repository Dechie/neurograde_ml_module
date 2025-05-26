import sys
for ab in sys.stdin:
    a,b=map(int,ab.split(" "))
    if a>=b:
        for g in range(b,0,-1):
            if a%g==0 and b%g==0:
                gcd = g
                break
        for l in range(1,b+1):
            if (a*l)%b == 0:
                lcm = a*l
                break
    else:
        for g in range(a,0,-1):
            if a%g==0 and b%g==0:
                gcd = g
                break
        for l in range(1,a+1):
            if (b*l)%a == 0:
                lcm = b*l
                break
    print gcd,lcm