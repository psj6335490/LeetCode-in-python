import numpy as np

arr=[3,34,4,12,5,2]
target=9

def rec_opt(arr,tar,i):
    if arr[i]==tar:
        return [i]
    elif i>1:
        result1=rec_opt(arr,tar-arr[i],i-1)
        result2 = rec_opt(arr, tar, i - 1)
        if result1:
            return [i,*result1]
        elif result2:
            return [ *result2]
        else:
            return None
    if i==0:
        return None

def dp_opt(arr,tar):
    opt=np.zeros((len(arr),tar+1),dtype=bool)
    opt[:,0]=True
    opt[0,arr[0]]=True
    for i in range(1,len(arr)):
        for j in range(1,tar+1):
            if opt[i-1][j]==True:
                opt[i][j]=True
            elif j>=arr[i] and opt[i-1][j-arr[i]]==True:
                 opt[i][j] = True
    x,y=opt.shape
    return opt[x-1,y-1]


if __name__ == '__main__':
    # result=rec_opt(arr,9,5)
    result = dp_opt(arr, 9)
    print(result)


