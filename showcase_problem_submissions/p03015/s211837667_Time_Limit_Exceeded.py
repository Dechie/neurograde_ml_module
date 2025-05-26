p = 10**9+7
def pow(x,m):
    if m==0:
        return 1
    if m==1:
        return x
    if m%2==0:
        return (pow(x,m//2)**2)%p
    else:
        return (x*pow(x,(m-1)//2)**2)%p
def f(i):
    tot = 0
    for k in range(i+1):
        tot = (tot + (((2**k)%p)*A[i]*B[k]*B[i-k])%p)%p
    return tot
L = input().strip()
N = len(L)
A = [1 for _ in range(N)]
for i in range(2,N):
    A[i] = (i*A[i-1])%p
B = [1 for _ in range(N)]
B[N-1] = pow(A[N-1],p-2)
for i in range(N-2,1,-1):
    B[i] = (B[i+1]*(i+1))%p
C = [0 for _ in range(N)]
for i in range(1,N):
    if L[i-1]=="1":
        C[i] = C[i-1]+1
    else:
        C[i] = C[i-1]
tot = 0
for i in range(N):
    if L[i]=="1":
        tot = (tot+f(N-1-i)*2**C[i])%p
if L[N-1]=="0":
    tot = (tot+2**C[N-1])%p
else:
    tot = (tot+2**(C[N-1]+1))%p
print(tot)