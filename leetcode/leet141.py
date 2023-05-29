
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head: ListNode) :
        if head is None or head.next is None:
            return None

        slow,fast=head,head

        index=None
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow==fast:
                index=slow
                break

        if not index:
            return None

        index1=head
        index2=index

        while index1!=index2:
            index1=index1.next
            index2=index2.next
        return index2

def test_has_cycle():
    # create linked list 1->2->3->4->5
    head = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(4)
    node5 = ListNode(5)
    head.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    # print(Solution().detectCycle(head))

    # create linked list 1->2->3->4->5->3 (cycle)
    node5.next = node3
    print(Solution().detectCycle(head))


if __name__ == '__main__':
    test_has_cycle()
