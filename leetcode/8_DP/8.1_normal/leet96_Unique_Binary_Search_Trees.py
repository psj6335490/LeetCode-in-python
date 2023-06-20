class Solution:
    def numTrees(self, n: int) -> int:
        if n<3: return n
        dp=[0 for _ in range(n+1)]
        dp[0] = 1
        dp[1]=1
        dp[2]=2
        for i in range(3,n+1):
            for j in range(i):
                dp[i]+=dp[j]*dp[i-1-j]
        return dp[-1]



if __name__ == '__main__':
    result = Solution().numTrees(3)
    print(result)