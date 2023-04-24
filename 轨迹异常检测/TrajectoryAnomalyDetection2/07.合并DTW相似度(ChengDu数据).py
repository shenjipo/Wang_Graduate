import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import heapq
import torch
from test2 import cal_dtw

inner_file_path = './data/ChengDu/data3/inners.txt'
outlier_file_path = './data/ChengDu/data3/outliers.txt'
# 5 3 5
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

dtw_array = [[0] * len(traj_list) for _ in range(len(traj_list))]

for index1, item1 in enumerate(traj_list):
    for index2, item2 in enumerate(traj_list):
        print(index1, index2, len(traj_list))
        if index1 != index2 and dtw_array[index1][index2] == 0:
            temp = cal_dtw(item1, item2)
            dtw_array[index1][index2] = temp
            dtw_array[index2][index1] = temp

for index1, item1 in enumerate(traj_list):
    scores = []
    # if index1 > 10:
    #     break
    print(index1, len(traj_list))
    for index2, item2 in enumerate(traj_list):
        if index1 != index2:
            temp_score = []
            # 计算dtw相似度
            temp_score.append(dtw_array[index1][index2])

            temp_score.append(index2)

            scores.append(copy.deepcopy(temp_score))
    # print(scores)
    scores.sort(key=lambda x: abs(x[0]), reverse=True)

    for i in range(5):
        votes[int(np.array(scores)[i, 1])] += 1

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
