class Solution:
    def climbStairs(self, n: int) -> int:
        if n<3 :return n
        dp=[0 for _ in range(n+1)]
        dp[1]=1
        dp[2]=2

        for i in range(3,n+1):
            dp[i]=dp[i-1]+dp[i-2]
        return dp[-1]


if __name__ == '__main__':
    result = Solution().climbStairs(3)
    print(result)