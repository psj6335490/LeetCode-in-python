class Solution:
    def twoSum(self, nums, target) :
        #method1:time consumer for .index
        # for i in range(len(nums)):
        #     if target-nums[i] in nums[i+1:]:
        #         return [i,nums[i+1:].index(target-nums[i])+i+1]
        # return []

        #only once
        dict = {}
        for i in range(len(nums)):
            if target - nums[i] not in dict:
                dict[nums[i]] = i
            else:
                return [dict[target - nums[i]], i]

print(Solution().twoSum([2,7,11,15],9))