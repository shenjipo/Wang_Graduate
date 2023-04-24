import numpy as np
import copy
import torch
from fastdtw import fastdtw
import time
import random
from test2 import cal_dtw

random.seed(5)
np.random.seed(5)
# 1.读取轨迹文件 并且准备好实际的结果与预测的结果
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

# 2. 轨迹预处理(根据DTW相似度以及DBA算法生成合成轨迹)
# 先随机挑选出30条轨迹计算他们之间的相似度，然后根据投票，选出相似度最高的20条轨迹
# 参数在这里 参数在这里 参数在这里 参数在这里 参数在这里 参数在这里 参数在这里 参数在这里
select_traj_number = 30
dba_traj_number = 20
select_traj_index = random.sample(range(0, len(traj_list)), select_traj_number)

DBA_traj_list = []
for index in select_traj_index:
    DBA_traj_list.append(copy.deepcopy(traj_list[index]))

votes = [0] * len(DBA_traj_list)
dtw_array = [[0] * len(traj_list) for _ in range(len(traj_list))]
dtw_average = []

start_votes = time.time()
for index1, item1 in enumerate(DBA_traj_list):
    for index2, item2 in enumerate(DBA_traj_list):
        print(index1, index2, len(DBA_traj_list))
        if index1 != index2 and dtw_array[index1][index2] == 0:
            distance, path = fastdtw(item1, item2, dist=lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]))
            # distance = cal_dtw(item1, item2)
            dtw_array[index1][index2] = distance
            dtw_array[index2][index1] = distance

for index1, item1 in enumerate(DBA_traj_list):
    scores = []
    print(index1, len(DBA_traj_list))
    for index2, item2 in enumerate(DBA_traj_list):
        if index1 != index2:
            temp_score = []
            # 计算dtw相似度
            temp_score.append(dtw_array[index1][index2])
            temp_score.append(index2)
            scores.append(copy.deepcopy(temp_score))
    scores.sort(key=lambda x: abs(x[0]), reverse=False)
    for i in range(dba_traj_number):
        votes[int(np.array(scores)[i, 1])] += 1

votes = torch.tensor(votes)
_, anomaly_index = torch.topk(votes, dba_traj_number)

end_votes = time.time()
open('./res/time/A_1_dtw.txt', mode='w').writelines(str(round(end_votes - start_votes, 2)) + '\n')
print('投票法运行时间{}'.format(end_votes - start_votes))
# 获取正常轨迹的均值DTW
for index1 in anomaly_index:
    for index2 in anomaly_index:
        if index1 != index2:
            temp = dtw_array[index1][index2]
            dtw_average.append(copy.deepcopy(temp))

# 索引映射
save_index = []
for value in anomaly_index:
    save_index.append(str(select_traj_index[value]))

open('./res/ATDC/data1.txt', mode='w').writelines(str(sum(dtw_average) // len(dtw_average)) + '\n')
open('./res/ATDC/data1.txt', mode='a').writelines(' '.join(save_index) + '\n')
