from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from matplotlib import pyplot as plt

s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
distance, paths = dtw.warping_paths(s1, s2
                                    # , window=25
                                    # , psi=2
                                    )
# print(distance)
# best_path = dtw.best_path(paths)  # 最短路径
# dtwvis.plot_warpingpaths(s1, s2, paths, best_path)  # 制图
# plt.show()
import random

print(random.sample(range(0, 5), 5))
