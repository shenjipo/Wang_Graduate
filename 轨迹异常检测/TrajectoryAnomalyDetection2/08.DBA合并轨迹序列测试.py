import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import heapq
import torch
from dtaidistance import dtw
from DTW_utils.utils import performDBA
import matplotlib.pyplot as plt

inner_file_path = './data/ATDC/data1/inners.txt'
outlier_file_path = './data/ATDC/data1/outliers.txt'

inner_list = []
new_inner_list = []
outlier_list = []
traj_list = []
with open(inner_file_path) as trajs:
    for index, line in enumerate(trajs):
        line = eval(line)
        inner_list.append(line)
        traj_list.append(line)
        temp = [np.array(line)[:, 0], np.array(line)[:, 1]]
        new_inner_list.append(np.array(temp))

with open(outlier_file_path) as trajs:
    for index, line in enumerate(trajs):
        line = eval(line)
        outlier_list.append(line)
        traj_list.append(line)

average_series = performDBA(new_inner_list)
print(average_series)

for index, traj in enumerate(inner_list):
    traj = np.array(traj)
    plt.scatter(traj[:, 0], traj[:, 1])

plt.scatter(average_series[0], average_series[1], color='r')
plt.show()
