from typing import List

class Solution:
    def helper(self, nums: List[int]) -> int:
        dp = [0 for _ in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = max(nums[1], dp[0])
        if len(nums) > 2:
            for i in range(2, len(nums)):
                dp[i] = max(nums[i] + dp[i - 2], dp[i - 1])

        return dp[len(nums) - 1]

    def rob(self, nums: List[int]) -> int:
        if len(nums)==0 :return 0
        if len(nums)==1:return nums[0]
        if len(nums)==2:return max(nums[0],nums[1])
        # 最终问题的解有可能3种:
        # 1.包含front,就不能包含back,等价于
        nums1=self.helper(nums[0:-1])
        # 2.包含back,就不能包含front,等价于
        nums2 = self.helper(nums[1:])
        # 3.都不包含,以上两种可能已经处理了

        return max(nums1,nums2)


if __name__ == '__main__':
    nums =[1,2,3,1]
    result = Solution().rob(nums)
    print(result)