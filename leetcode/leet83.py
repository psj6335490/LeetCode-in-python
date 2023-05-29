class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteDuplicates(self, head) :
        if head is None:
            return head
        slow, fast = head, head
        while fast  :
            if fast.val!=slow.val:
                slow.next=fast
                slow=fast
            fast=fast.next
        slow.next=fast
        return head

def test_has_cycle():
    # create linked list 1->2->3->4->5
    head = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(3)
    node5 = ListNode(5)
    head.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    return (Solution().deleteDuplicates(head))

    # create linked list 1->2->3->4->5->3 (cycle)
    # node5.next = node3
    # print(Solution().detectCycle(head))

if __name__ == '__main__':
    result=test_has_cycle()

    while result:
        print(result.val)
        result=result.next
