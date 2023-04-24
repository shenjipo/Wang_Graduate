import collections
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score


def loadData(path):
    with open(path, 'r') as f:
        routes = f.readlines()
    return routes


def findSourceDestination(routes):
    X = []
    for i in range(len(routes)):
        route = np.array(eval(routes[i]))
        X.append(route[0])
        X.append(route[-1])
    X = np.array(X)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    res = kmeans.cluster_centers_
    return res


def sortPointsFromS2D(route, res):
    # res[0] is the S point
    if sum((res[0] - route[0]) ** 2) <= sum((res[1] - route[0]) ** 2):
        return route
    else:
        return route[::-1]


def positionOfLine(A, B, C):
    Ax, Ay, Bx, By, X, Y = A[0], A[1], B[0], B[1], C[0], C[1]
    position = np.sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
    if position >= 0:
        return 1
    else:
        return -1


def calAngle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    # cal the angle between side ab and bc
    ba = a - b
    bc = c - b

    t1 = np.dot(ba, bc)
    t2 = np.linalg.norm(ba)
    t3 = np.linalg.norm(bc)

    if np.isnan(t1) or np.isnan(t2) or (t2 * t3) == 0:
        return 0
    if np.isnan(t3):
        return 180

    else:
        cosine_angle = t1 / (t2 * t3)
        if cosine_angle > 1:
            cosine_angle = 1
        if cosine_angle < -1:
            cosine_angle = -1
        angle = np.arccos(cosine_angle)
        p = positionOfLine(a, b, c)
        if p == 1:
            return angle * 180 / np.pi
        else:
            return 360 - angle * 180 / np.pi


def extractFearureMiPo(route, res, k):
    # route is a trajectory
    # res=[S,D], S and D is the scource and destination point,respectively.
    # k for the number of bins

    if k <= 0:
        return "error: k should be more than 0"

    # cal angles
    S, M, D = res[0], sum(res) / 2, res[1]

    # distribute points into bucket
    bin_angle_range = 180 / k  # correct

    feature_ponit = [0, 0]
    bucket = {i: [] for i in range(k)}
    bucket_dist = {i: 0 for i in range(k)}

    bucket[0].append(0)
    for i in range(1, len(route) - 1):
        angles = calAngle(S, M, route[i])
        angles_bin = 0
        if angles <= 180:
            angles_bin = angles // bin_angle_range
            feature_ponit[0] += 1
        else:
            angles_bin = (360 - angles) // bin_angle_range
            feature_ponit[1] += 1
        idx = int(angles_bin)
        if idx > k - 1:
            idx = k - 1
        bucket[idx].append(i)
        bucket_dist[idx] += np.sqrt(np.sum((route[i] - M) ** 2))  # distance fearure
    bucket[k - 1].append(-1)

    # distance feature
    feature_dist = []
    for j in range(len(bucket)):
        avg_dst = 0
        a = len(bucket[j])
        if a > 0:
            avg_dst = bucket_dist[j] / a
        else:
            if len(feature_dist) > 0:
                avg_dst = feature_dist[-1]
            else:
                avg_dst = np.sqrt(np.sum((S - M) ** 2))
        feature_dist.append(avg_dst)

    feature = feature_dist + feature_ponit
    return feature


# 读取轨迹数据
# data1 30 48
# data2 20 25
# data3 20 30
path_inner = "./data/ChengDu/data2/inners2.txt"
path_outlier = "./data/ChengDu/data2/outliers2.txt"

outliers = loadData(path_outlier)
inners = loadData(path_inner)

label = [1] * len(outliers) + [0] * len(inners)
data = outliers + inners
print('dataset####', len(data))

# find S-D
SD_points = findSourceDestination(data)

time_e0 = time.time()

# MiPo for feature extraction
k = 20  # for bins
features = []
for i in range(len(data)):
    p1_resorted = np.array(eval(data[i]))
    p1_resorted = np.concatenate((np.array([SD_points[0]]), p1_resorted, np.array([SD_points[-1]])), axis=0)
    feature = extractFearureMiPo(p1_resorted, SD_points, k)
    features.append(feature)
#     time_e1 = time.time()
#     print(time_e1 - time_e0)  # time for feature

# min-max scaling
features = np.array(features)
features = (np.array(features) - np.min(features, axis=0)) / (np.max(features, axis=0) + 0.0001)

# LOF detector
n_neighbors = 25  # for LOF
clf = LocalOutlierFactor(n_neighbors)
OutlierScore = -clf.fit_predict(features)

pred_label = [item + 1 if item == -1 else item for item in OutlierScore.tolist()]
print(pred_label)
tp = [pred_label[i] == 1 and label[i] == 1 for i in range(len(label))].count(True)
fn = [pred_label[i] == 0 and label[i] == 1 for i in range(len(label))].count(True)
fp = [pred_label[i] == 1 and label[i] == 0 for i in range(len(label))].count(True)
tn = [pred_label[i] == 0 and label[i] == 0 for i in range(len(label))].count(True)
p = 0 if (tp + fp) == 0 else tp / (tp + fp)
r = 0 if (tp + fn) == 0 else tp / (tp + fn)
f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
print('p={}, r={}, f1={} '.format(p, r, f1))