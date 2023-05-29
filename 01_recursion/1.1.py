# 获得子问题
# 合并结果
# 定义出口

#方法 暴力穷举 分治法 逐步迭代 随机化

#1.递归全排列
def premutation(elements):
    if len(elements)==1:
        return [elements]
    head=elements[0]

    remain_parts_results=premutation(elements[1:])
    return [r[:i]+[head]+r[i:]for i in range(len(elements)) for r in remain_parts_results ]


#2.归并排序
def merge_sort(elements):
    if len(elements)==1: return elements

    mid=len(elements)//2

    left_part,right_part=merge_sort(elements[:mid]),merge_sort(elements[mid:])

    sort_result=[]

    while left_part and right_part:
        left_head=left_part[0]
        right_head=right_part[0]
        if left_head<right_head:
            sort_result.append(left_head)
            left_part.pop(0)
        else:
            sort_result.append(right_head)
            right_part.pop(0)

    sort_result+=(left_part or right_part)
    return sort_result

#3.快速排序
import random
def quick_sort(elements):
    if len(elements)==0: return []
    # if not elements: return []
    pivot=random.choice(elements)
    return quick_sort([e for e in elements if e<pivot]) +[e  for e in elements if e==pivot]+quick_sort([e  for e in elements if e>pivot])


#4.Graph Traverse

simple_graph={
    'A':'B C D'.split(),
    'B':'A'.split(),
    'C':'A E'.split(),
    'D':'A'.split(),
    'E':'F C G W'.split(),
    'W':'E'.split(),
    'F':'H I E'.split(),
    'G':'H E'.split(),
    'H':'I F G'.split(),
}


# def traverse(start,connection):
#     if not connection: return  start
#     sub_graph=connection[start]
#
#     connection.pop(start)
#     result=[start]
#
#     for r in sub_graph:
#         if r not in connection:continue
#         result+=


import networkx as nx
nx.draw(nx.Graph(simple_graph),with_labels=True)


#5.Edit Distance
from functools import lru_cache

solution={}
@lru_cache(maxsize=2**10)
def edit_distance(str1,str2):
    if not str1:return len(str2)
    if not str2:return len(str1)

    condidate=[]
    if str1[-1]==str2[-1]:
        condidate.append((edit_distance(str1[0:-1],str2[0:-1]),'FORWARD '))
    else:
        condidate.append((edit_distance(str1[0:-1], str2[0:-1])+1,'SUB {}->{}'.format(str1[-1],str2[-1])))
    condidate.append((edit_distance(str1[0:-1], str2)+1,'DEL ->{}'.format(str1[-1])))
    condidate.append((edit_distance(str1, str2[0:-1])+1,'ADD ->{}'.format(str2[-1])))

    min_distance,operation=min(condidate,key=lambda x:x[0])
    solution[(str1,str2)]=operation
    return min_distance

def parse_solution(str1,str2):


    operations={
        'FORWARD':(-1,-1),
        'SUB':(-1,-1),
        'DEL':(-1,None),
        'ADD':(None,-1),
    }
    # if len(str1)==len(str2)==1:return [solution[(str1,str2)]]
    if len(str1)==0 or len(str2)==0:return []

    operation=solution[(str1,str2)]
    op=operation.split()[0]
    x1,x2=operations[op]

    return [operation]+parse_solution(str1[:x1],str2[:x2])


#6 休息10分钟
from  tqdm import tqdm_notebook
# from tqdm import tqdm
# import time
# for i in tqdm(range(60*10)):
#     time.sleep(1)
#     pass



#7过滤重复文字
# numbers=[random.randint(0,10) for _ in range(100)]
# numbers=[0, 10, 1, 0, 1, 5, 5, 1, 8, 3, 1, 2, 6, 8, 9, 5, 9, 5, 8, 2, 0, 3, 5, 2, 0, 3, 9, 2, 0, 5, 3, 5, 7, 0, 8, 3, 8, 9, 5, 10, 6, 2, 3, 8, 6, 4, 2, 2, 7, 9, 4, 0, 7, 4, 8, 5, 3, 3, 7, 6, 2, 2, 4, 7, 8, 2, 7, 4, 10, 7, 3, 10, 6, 0, 8, 1, 4, 10, 8, 4, 3, 6, 6, 5, 8, 9, 0, 7, 0, 1, 8, 1, 8, 0, 2, 5, 4, 6, 4, 7]
numbers='老子打打打死你你'
def filter_pattern(elements):
    if not elements:return []
    for i in range(len(elements)//2,0,-1):
        remain_elements=elements[i:]
        if elements[:i]==remain_elements[:i]:
            return filter_pattern(remain_elements)
    return [elements[0]]+filter_pattern(elements[1:])


#8 flatten
data_list=(1,(((2,3,4))),((5,6,7,),(8,9,10)),11,12,(13,14,15,16,17,18),19,20,21)
def flatten1(elements):

    if not elements: return []
    if isinstance(elements,int):return [elements]
    return flatten1(elements[0])+flatten1(elements[1:])

def flatten2(elements):

    result=[]
    if isinstance(elements,int):return [elements]
    for i in elements:
        result+=flatten2(i)

    return result


if __name__ == '__main__':
    result=premutation([12 , "sf",5])

    # print(merge_sort([12,3,2,4,]))
    # print(quick_sort([12,2,2,4,]))
    # traverse('D',simple_graph)
    # result=edit_distance('123des5','desk2')
    # result=parse_solution('123des5','desk2')

    # result=min([(9,11),(1,12),(3,13),(-1,18),(19,7)],key=lambda x:x[1])
    # result=flatten1(data_list)
    # result=flatten2(data_list)
    # result=filter_pattern(numbers)

    print(result)
    pass