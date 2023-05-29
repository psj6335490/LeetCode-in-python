import numpy as np

arr=[1,2,4,1,7,8,3]

def rec_opt(arr,i):
    if i==0:
        return arr[0]
    if i==1:
        return max(arr[0],arr[1])
    a=arr[i]+rec_opt(arr,i-2)
    b=rec_opt(arr,i-1)
    return max(a,b)

def dp_opt(arr):
    opt=np.zeros(len(arr))
    opt[0]=arr[0]
    opt[1]=max(arr[0],arr[1])

    for i in range(2,len(arr)):
        a=arr[i]+opt[i-2]
        b=opt[i-1]
        opt[i]=max(a,b)
    return opt[-1]


if __name__ == '__main__':
    # result=rec_opt(arr,6)
    result=dp_opt(arr)
    print(result)