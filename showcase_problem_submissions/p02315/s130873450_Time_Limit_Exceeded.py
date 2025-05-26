import copy
def main():
    [N,W]=input().split(" ")
    '''read from keyboard and put all the item to box'''
    N=int(N)
    W=int(W)
    box=[]
    for i in range(N):
        [vi,wi]=input().split(" ")
        box.append([float(vi),float(wi)])
        
    assemblebox=[]
    '''creat a set of full permutation of input. using recursion'''
    assemblebox=permutations(N,box,assemblebox,[])

    maximum=0
    '''optimization'''
    for i in range(len(assemblebox)):
        weight=0
        value=0
        for j in range(N):
            weight=weight+assemblebox[i][j][1]
            if(weight>W):
                break
            value=value+assemblebox[i][j][0]
        if(maximum<value):
            maximum=value
            
    print(int(maximum))
    '''output'''
        

    return 0

def permutations(N,box,assemblebox,record):
    for i in range(N):
        Np=N
        boxp=copy.deepcopy(box)
        recordp=copy.deepcopy(record)
        recordp.append(box[i])
        boxp.pop(i)
        Np=Np-1
        if(Np==0):
            assemblebox.append(recordp)
        else:
            result=permutations(Np,boxp,assemblebox,recordp)
    return assemblebox

main()