import numpy as np
from utils import calAngle, calDisByeuclidean
import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor

inner_file_path = './data/ChengDu/data1/inners1.txt'
outlier_file_path = './data/ChengDu/data1/inners1.txt'
inner_list = []
traj_list = []

# 读取轨迹数据
with open(inner_file_path) as file:
    for index, line in enumerate(file):
        line = eval(line)
        traj_list.append(line)

with open(outlier_file_path) as file:
    for index, line in enumerate(file):
        line = eval(line)
        traj_list.append(line)
print(len(traj_list))

start = [7, 82]
end = [119, 40]
middle = [int((start[0] + end[0]) // 2), int((start[1] + end[1]) // 2)]

bins = 5
interval = 180 // bins

features = []
for traj in traj_list:
    dis = [[] for _ in range(bins)]
    for point in traj:
        temp = calAngle(start, middle, point)
        temp = 360 - temp if temp > 180 else temp
        temp = 179.9 if temp == 180.0 else temp
        index = int(temp // interval)
        dis[index].append(calDisByeuclidean(middle, point))

    feature = [np.mean(item) if len(item) >= 1 else 0 for item in dis]
    features.append(copy.deepcopy(feature))

print(features[0])
print(features[-1])
n_neighbors = 5  # for LOF
clf = LocalOutlierFactor(n_neighbors, contamination=0.15)
OutlierScore = clf.fit_predict(features)
OutlierScore = OutlierScore.tolist()
print(OutlierScore)
print(clf.negative_outlier_factor_)
