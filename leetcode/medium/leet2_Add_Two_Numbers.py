from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy=ListNode(0)
        p=dummy
        carry=0
        while l1 or l2:
            if not l1:
                l1=0
            if not l2:
                l2=0

            val=(l1.val if l1 else 0)+(l2.val if l2 else 0)+carry
            p.next=ListNode(val%10)
            carry=val//10
            p=p.next
            if l1:
                l1=l1.next
            if l2:
                l2=l2.next

        if carry==1:
            p.next=ListNode(carry)

        return dummy.next

if __name__ == '__main__':
    a=[9, 9, 9, 9, 9, 9, 9]
    b=[9, 9, 9, 9]
    l1 = ListNode(0)
    temp=l1
    for i in a:
        temp.next = ListNode(i)
        temp=temp.next
    l1=l1.next

    l2 = ListNode(0)
    temp = l2
    for i in b:
        temp.next = ListNode(i)
        temp = temp.next

    l2=l2.next


    result=Solution().addTwoNumbers(l1,l2)
    while result:
        print(result.val)
        result=result.next



