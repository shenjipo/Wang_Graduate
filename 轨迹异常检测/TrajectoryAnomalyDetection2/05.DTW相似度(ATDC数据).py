import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import heapq
import torch
from dtaidistance import dtw

inner_file_path = './data/ATDC/data3/inners.txt'
outlier_file_path = './data/ATDC/data3/outliers.txt'

inner_list = []
outlier_list = []
traj_list = []
with open(inner_file_path) as trajs:
    for index, line in enumerate(trajs):
        line = eval(line)
        inner_list.append(line)
        traj_list.append(line)

with open(outlier_file_path) as trajs:
    for index, line in enumerate(trajs):
        line = eval(line)
        outlier_list.append(line)
        traj_list.append(line)

print(len(outlier_list), len(inner_list))
label = [0] * len(inner_list) + [1] * len(outlier_list)
diff_list_x = []
inter_list_x = []
diff_list_y = []
inter_list_y = []

features = []

votes = [0] * len(traj_list)
end = int(len(traj_list) * 0.1)

xdtw_array = [[0] * len(traj_list) for _ in range(len(traj_list))]
ydtw_array = [[0] * len(traj_list) for _ in range(len(traj_list))]

for index1, item1 in enumerate(traj_list):
    for index2, item2 in enumerate(traj_list):
        print(index1, index2, len(traj_list))
        if index1 != index2 and xdtw_array[index1][index2] == 0:
            traj1_x = np.array(item1)[:, 0]
            traj1_y = np.array(item1)[:, 1]
            traj2_x = np.array(item2)[:, 0]
            traj2_y = np.array(item2)[:, 1]
            xdtw_distance = dtw.distance(traj1_x, traj2_x)
            xdtw_array[index1][index2] = xdtw_distance
            xdtw_array[index2][index1] = xdtw_distance

            ydtw_distance = dtw.distance(traj1_y, traj2_y)
            ydtw_array[index1][index2] = ydtw_distance
            ydtw_array[index2][index1] = ydtw_distance

for index1, item1 in enumerate(traj_list):
    scores = []
    # if index1 > 10:
    #     break
    print(index1, len(traj_list))
    for index2, item2 in enumerate(traj_list):
        if index1 != index2:
            temp_score = []
            # 计算x轴上的dtw相似度
            temp_score.append(xdtw_array[index1][index2])
            # 计算y轴上的dtw相似度
            temp_score.append(ydtw_array[index1][index2])
            temp_score.append((temp_score[0] + temp_score[1]) / 2)
            temp_score.append(index2)

            scores.append(copy.deepcopy(temp_score))
    # print(scores)
    scores.sort(key=lambda x: abs(x[2]), reverse=True)

    for i in range(13):
        votes[int(np.array(scores)[i, 3])] += 1

# 0-15正常 16 17 18 19 20 异常
print(votes)
votes = torch.tensor(votes)

_, anomaly_index = torch.topk(votes, len(outlier_list))
anomaly_index = anomaly_index.detach().cpu().tolist()
pred_label = [1 if item in anomaly_index else 0 for item in range(len(traj_list))]
print(anomaly_index)
print(pred_label)


tp = [pred_label[i] == 1 and label[i] == 1 for i in range(len(label))].count(True)
fn = [pred_label[i] == 0 and label[i] == 1 for i in range(len(label))].count(True)
fp = [pred_label[i] == 1 and label[i] == 0 for i in range(len(label))].count(True)
tn = [pred_label[i] == 0 and label[i] == 0 for i in range(len(label))].count(True)
p = 0 if (tp + fp) == 0 else tp / (tp + fp)
r = 0 if (tp + fn) == 0 else tp / (tp + fn)
f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
print('p={}, r={}, f1={} '.format(p, r, f1))


