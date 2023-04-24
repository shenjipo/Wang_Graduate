import matplotlib.pyplot as plt
import numpy as np

innerPath = './data/ChengDu/data2/inners.txt'
outlierPath = './data/ChengDu/data2/outliers.txt'
traj_list = []

inner_list = []
outlier_list = []
with open(innerPath) as traj:
    for index, line in enumerate(traj):
        line = eval(line)
        inner_list.append(line)
        traj_list.append(line)

with open(outlierPath) as traj:
    for index, line in enumerate(traj):
        line = eval(line)
        outlier_list.append(line)
        traj_list.append(line)

for index, traj in enumerate(inner_list):
    if index ==12:
        traj = np.array(traj)
        plt.scatter(traj[:, 0], traj[:, 1])

# for index, traj in enumerate(outlier_list):
#     traj = np.array(traj)
#     plt.scatter(traj[:, 0], traj[:, 1], color='r')
plt.show()
