import numpy as np

class Solution:
    def integerBreak(self, n: int) -> int:
        opt = [0 for _ in range(n + 1)]

        opt[1] = 1
        opt[2] = 1
        for i in range(3, n + 1):
            for j in range(1, i):
                a=j*(i-j)
                b=j*opt[i-j]
                opt[i]=max(opt[i],a,b)

        return opt[-1]

        # 多填写了一半
        def integerBreak1(self, n: int) -> int:
            opt = np.zeros((n + 1, n + 1), dtype=int)
            opt[:, 1] = 1
            opt[:, 2] = 1

            for i in range(1, n + 1):
                for j in range(3, n + 1):
                    # 多填写了一半
                    if j < i:
                        opt[i, j] = opt[i - 1, j]
                    else:
                        a = i * opt[i, j - i]
                        b = i * (j - i)
                        c = opt[i - 1, j]
                        opt[i, j] = max(a, b, c)

            r, c = opt.shape
            return opt[r - 1, c - 1]
            # return opt


if __name__ == '__main__':
    result=Solution().integerBreak(10)
    print(result)