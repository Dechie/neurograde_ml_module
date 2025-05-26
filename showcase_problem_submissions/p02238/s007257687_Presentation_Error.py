n=0
t=1
def adjacent_matrix():
    global n
    n=int(input())
    v=[]
    array=[]
    for i in range(n):
        zero=[0 for i in range(n)]
        a=list(map(int,input().split()))
        a=a[2:]
        for j in a:
            zero[j-1]=1
        array.append(zero)
    return array
def dfs(array,i,d,f):
    global t
    if i == n:
        return
    else:
        if(d[i] == 0):
            d[i] = t
            t += 1
            for (j,x) in enumerate(array[i]):
                if x == 1:
                    dfs(array,j,d,f)
            f[i] = t
            t += 1
    return 


def main():
    array=adjacent_matrix()
    print()
    d=[0 for i in range(n)]
    f=[0 for i in range(n)]
    for i in range(n):
        dfs(array,i,d,f)
    for i in range(n):
        print("{} {} {} ".format(i+1,d[i],f[i]))

if __name__ == "__main__":
    main()
