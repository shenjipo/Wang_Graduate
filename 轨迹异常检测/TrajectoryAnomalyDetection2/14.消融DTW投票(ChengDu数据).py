import numpy as np
import random
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import heapq
import torch
from dtaidistance import dtw
from DTW_utils.utils import performDBA
import matplotlib.pyplot as plt
from test2 import cal_dtw

random.seed(5)
np.random.seed(5)
# 1.读取轨迹文件 并且准备好实际的结果与预测的结果
inner_file_path = './data/ChengDu/data2/inners2.txt'
outlier_file_path = './data/ChengDu/data2/outliers2.txt'

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
#
# 随机选取N条正常的轨迹 并且计算他们彼此的平均dtw相似度
k = 25
normal_file_index = random.sample(range(0, len(inner_list)), k)
print(normal_file_index)
dtw_average = []
for index1 in normal_file_index:
    for index2 in normal_file_index:
        if index1 != index2:
            temp = cal_dtw(inner_list[index1], inner_list[index2])
            dtw_average.append(copy.deepcopy(temp))
average_dtw = sum(dtw_average) // len(dtw_average)
# 消融结束

label = [0] * len(inner_list) + [1] * len(outlier_list)
pred_label = [0] * (len(traj_list))

normal_list = []
# 初始的k条轨迹都是正常轨迹
for value in normal_file_index:
    pred_label[value] = 0
    temp = traj_list[value]
    temp = [np.array(temp)[:, 0], np.array(temp)[:, 1]]
    normal_list.append(np.array(temp))
# 生成初始的合成轨迹
DBA_list = performDBA(normal_list)
DBA_list = np.array(DBA_list).transpose(1, 0).tolist()

print('初始DBA轨迹生成结束')

# 可视化DBA轨迹
# plt.scatter(DBA_list[0], DBA_list[1], color='r')
# plt.show()

# 遍历剩余轨迹
# 设置阈值
# data1 3.3
# data2 2.1
# data3 2.5
threshold = 3
count_normal = k
for index, curr_traj in enumerate(traj_list):
    if index not in normal_file_index:
        cur_dtw = cal_dtw(DBA_list, curr_traj)
        if cur_dtw < threshold * average_dtw:
            pred_label[index] = 0
            # 正常轨迹 更新DBA轨迹
            series = []
            series.append(np.array(curr_traj).transpose(1, 0))
            series.append(np.array(DBA_list).transpose(1, 0))
            DBA_list = performDBA(series)
            DBA_list = np.array(DBA_list).transpose(1, 0).tolist()
            # 更新均值
            # 更新均值
            average_dtw = sum([cur_dtw, average_dtw * count_normal]) / (count_normal + 1)
            count_normal += 1
        else:
            # 异常轨迹
            pred_label[index] = 1
            cur_dtw = cal_dtw(DBA_list, curr_traj)
        print(cur_dtw, index, average_dtw)

# 计算指标
print(pred_label)
tp = [pred_label[i] == 1 and label[i] == 1 for i in range(len(label))].count(True)
fn = [pred_label[i] == 0 and label[i] == 1 for i in range(len(label))].count(True)
fp = [pred_label[i] == 1 and label[i] == 0 for i in range(len(label))].count(True)
tn = [pred_label[i] == 0 and label[i] == 0 for i in range(len(label))].count(True)
p = 0 if (tp + fp) == 0 else tp / (tp + fp)
r = 0 if (tp + fn) == 0 else tp / (tp + fn)
f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
print('p={}, r={}, f1={} '.format(p, r, f1))
