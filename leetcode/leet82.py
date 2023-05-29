class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
#(1)
# class Solution:
#     #头结点有可能被删除,这是要创建一个节点指向头结点,同时用快指针自己和next比较,pre只记录不操作
#     def deleteDuplicates(self, head) :
#         empty=ListNode(0)
#         empty.next=head
#         if head is None:
#             return head
#         pre, cur = empty, head
#         while cur :
#                 对比方法一和方法二:方法一是可以移动并且不等条件成立才判断,快结束时不判断
#                 方法二除了相等每次都判断,包含快结束
#                 应该是除了相等之外每次都判断
#             if cur.next and cur.val!=cur.next.val :
#                 if pre.next==cur:
#                     pre=cur
#                 else:
#                     pre.next=cur.next
#             cur=cur.next
#
#         if pre.next.next != cur:
#             pre.next = cur
#
#         return empty.next

#(2)
class Solution:
    #头结点有可能被删除,这是要创建一个节点指向头结点,同时用快指针自己和next比较,pre只记录不操作
    def deleteDuplicates(self, head) :
        empty=ListNode(0)
        empty.next=head

        pre, cur = empty, head
        while cur :
            while cur.next and cur.val==cur.next.val :
                cur = cur.next

            if pre.next==cur:
                pre=cur
            else:
                pre.next=cur.next
            cur=cur.next

        return empty.next

def test_has_cycle():
    # create linked list 1->2->3->4->5
    head = ListNode(1)
    node2 = ListNode(1)
    node3 = ListNode(2)
    node4 = ListNode(3)
    node5 = ListNode(3)
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
