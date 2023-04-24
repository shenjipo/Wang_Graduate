import sys
import numpy as np
from dtaidistance import dtw


def mydis(x, y):

    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def cal_dtw(s1, s2):

    ts_a, ts_b = np.array(s1), np.array(s2)
    m, n = len(ts_a), len(ts_b)

    cost = sys.maxsize * np.ones((m, n))

    # 初始化第一个元素
    cost[0, 0] = mydis(ts_a[0], ts_b[0])

    # 计算损失矩阵第一列
    for i in range(1, m):
        cost[i, 0] = cost[i - 1, 0] + mydis(ts_a[i], ts_b[0])

    # 计算损失矩阵第一行
    for j in range(1, n):
        cost[0, j] = cost[0, j - 1] + mydis(ts_a[0], ts_b[j])

    # 计算损失矩阵剩余元素
    for i in range(1, m):
        for j in range(1, n):
            choices = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
            cost[i, j] = min(choices) + mydis(ts_a[i], ts_b[j])

    return cost[-1, -1]


# if __name__ == '__main__':
#     ts_a = [1, 5, 8, 10, 56, 21, 32, 8]
#     ts_b = [1, 5, 8, 10, 23, 56, 21, 32, 8]
#     ts_c = [1, 3, 6, 9, 16, 29, 31, 32, 33]
#     t1 = [[1, 2], [3, 4]]
#     t2 = [[1, 2], [3, 4]]
#     print(cal_dtw(t1, t2))
#     # 调用cal_dtw_distance计算dtw相似度
#     # dtw_ab = cal_dtw(ts_a, ts_b)
#     # dtw_ac = cal_dtw(ts_a, ts_c)
#     #
#     # print(dtw_ab)
#     # print(dtw.distance(ts_a, ts_b))
#     # # print(dtw_ac)
