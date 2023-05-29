import time
a=[]
def hannuota(n,A,B,C):
    if n==1:a.append((A,'=>',C))

    else:
        hannuota(n-1,A,C,B)
        a.append((A, '=>', C))
        hannuota(n-1,B,A,C)

stat=time.time()
hannuota(20,'塔1','塔2','塔3')
end=time.time()
print(end-stat)

print(len(a))

# for i in a :
#     print(i)
