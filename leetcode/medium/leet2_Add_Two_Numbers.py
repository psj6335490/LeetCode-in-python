#Optional
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy=ListNode[0]
        p=dummy

        while l1 or l2:
            if not l1:
                l1=0
            if not l2:
                l2=0

            
