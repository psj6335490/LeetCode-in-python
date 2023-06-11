from typing import List

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m,n=len(matrix),len(matrix[0])
        if m==0 or n==0 or not matrix:return 0
        # dp[i][j]表示以(i,j)为右下角形成的正方形的最大边长是多少
        dp=[[0 for _ in range(n)] for _ in range(m)]

        max_edge=0
        for i in range(m):
            for j in range(n):
                if i==0 or j==0:
                    dp[i][j]=int(matrix[i][j])
                elif int(matrix[i][j])==1:
                    dp[i][j]=min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1
                max_edge=max(max_edge,dp[i][j])

        return max_edge**2





if __name__ == '__main__':
    oblist=[["0","1"],["1","0"]]
    result = Solution().maximalSquare(oblist)
    print(result)