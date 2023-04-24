import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import heapq

inner_file_path = './data/ATDC/data1/inners.txt'
outlier_file_path = './data/ATDC/data1/outliers.txt'

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


for index1, item1 in enumerate(traj_list):
    scores = []
    # if index1 > 10:
    #     break
    print(index1, len(traj_list))
    for index2, item2 in enumerate(traj_list):
        if index1 != index2:
            temp_score = []
            traj1_x = np.array(item1)[:, 0]
            traj1_y = np.array(item1)[:, 1]
            traj2_x = np.array(item2)[:, 0]
            traj2_y = np.array(item2)[:, 1]
            # 计算x轴上的交集与差集
            diff = list(set(traj1_x).difference(set(traj2_x)))
            diff1 = list(set(traj2_x).difference(set(traj1_x)))
            inter = list(set(traj1_x).intersection(set(traj2_x)))

            diff_list_x.append(len(diff) - len(diff1))
            inter_list_x.append(len(inter))

            temp_score.append(diff_list_x[-1] / inter_list_x[-1])
            # 计算y轴上的交集与差集
            diff = list(set(traj1_y).difference(set(traj2_y)))
            diff1 = list(set(traj2_y).difference(set(traj1_y)))
            inter = list(set(traj1_y).intersection(set(traj2_y)))
            diff_list_y.append(len(diff) - len(diff1))
            inter_list_y.append(len(inter))
            temp_score.append(diff_list_y[-1] / inter_list_y[-1])
            temp_score.append((temp_score[0] + temp_score[1]) / 2)
            temp_score.append(index2)

            scores.append(copy.deepcopy(temp_score))
    scores.sort(key=lambda x: abs(x[2]), reverse=False)

    scores = scores[:10]
    features.append(np.array(scores)[:, 2])



# n_neighbors = 5  # for LOF
# clf = LocalOutlierFactor(n_neighbors)
#
features = np.array(features)

features = list(map(lambda x: str(x.tolist()) + '\n', features))
open('./test1.txt', mode='w').writelines(features)
print(features)

# OutlierScore = -clf.fit_predict(features)
# pred_label = [item + 1 if item == -1 else item for item in OutlierScore.tolist()]
# # print(pred_label)
# print(clf.negative_outlier_factor_.tolist())
# # print(label)
#
# roc = roc_auc_score(label, OutlierScore)
# print('auc', roc)
#
# tp = [pred_label[i] == 1 and label[i] == 1 for i in range(len(label))].count(True)
# fn = [pred_label[i] == 0 and label[i] == 1 for i in range(len(label))].count(True)
# fp = [pred_label[i] == 1 and label[i] == 0 for i in range(len(label))].count(True)
# tn = [pred_label[i] == 0 and label[i] == 0 for i in range(len(label))].count(True)
# p = 0 if (tp + fp) == 0 else tp / (tp + fp)
# r = 0 if (tp + fn) == 0 else tp / (tp + fn)
# f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
# print('p={}, r={}, f1={} '.format(p, r, f1))
