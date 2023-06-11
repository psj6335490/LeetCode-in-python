from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums :return 0
        if len(nums)==1:return nums[0]
        dp=[0 for _ in range(len(nums))]
        dp[0]=nums[0]
        dp[1]=max(nums[1],dp[0])
        if len(nums)>2:
            for i in range(2,len(nums)):
                dp[i]=max(nums[i]+dp[i-2],dp[i-1])

        return dp[len(nums)-1]


if __name__ == '__main__':
    nums =[0,0]
    result = Solution().rob(nums)
    print(result)