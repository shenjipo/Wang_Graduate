from test2 import cal_dtw
import numpy as np
import random

from dtaidistance import dtw, dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis

# s1 = [[1, 2], [3, 4], [5, 6], [7, 8]]
# s2 = [[1, 3], [3, 4], [5, 7], [8, 9]]
# s3 = [[2, 1], [3, 5], [4, 9], [8, 7]]
# s4 = [[3, 5], [1, 5], [6, 7], [8, 9]]
# traj_list = [s1, s2, s3, s4]
# print(cal_dtw(s1, s2))
# dtw_array = [[0] * len(traj_list) for _ in range(len(traj_list))]
# dtw_average = []

# for index1, item1 in enumerate(traj_list):
#     for index2, item2 in enumerate(traj_list):
#         print(index1, index2, len(traj_list))
#         if index1 != index2 and dtw_array[index1][index2] == 0:
#             temp = cal_dtw(item1, item2)
#             dtw_array[index1][index2] = temp
#             dtw_array[index2][index1] = temp
# print(dtw_array)
s1 = np.array([[0, 0],
               [0, 2],
               [2, 2],
               [4, 2],
               [4, 4],
               [6, 4],
               [7, 5],
               [8, 7],
               [9, 9]
               ], )
s2 = np.array([[0, 0],
               [2, 1],
               [4, 2],
               [5, 5],
               [7, 5],
               [8, 7],
               [9, 9]], )
# 0+2+1

print(cal_dtw(s1.tolist(), s2.tolist()))
distance, path = dtw_ndim.warping_paths(s1, s2)
from matplotlib import pyplot as plt
print(path)
dtwvis.plot_warpingpaths(s1, s2, np.array(path))  # 制图
plt.show()