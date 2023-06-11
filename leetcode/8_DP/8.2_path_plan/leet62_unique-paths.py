class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        s=1
        b=1
        for i in range(m,m+n-1):
            s*=i
        for j in range(2,n):
            b*=j
        return  s/b


if __name__ == '__main__':

    result = Solution().uniquePaths(7,3)
    print(result)