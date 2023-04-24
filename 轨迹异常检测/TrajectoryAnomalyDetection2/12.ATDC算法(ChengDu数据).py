import pandas as pd
import math

border1 = 2.5
border2 = border1 * 2
theta = [border2, border1, -border1, -border2]
phi = [-0.05, 0.05]
kn = 10
# data3 2   0.4651
# data4 1.0 0.48
# data5 8.0 0.54
fileIndex = 2
traj_path = './data/ChengDu/data{}/trajs.txt'.format(fileIndex)

cell_dict = {}
true_label_dict = {}
pred1_label_dict = {}
pred2_label_dict = {}

iso_df = pd.DataFrame(columns=('index', 'diff', 'inter', 'ratio', 'pred'))
iso_df2 = pd.DataFrame(columns=('index', 'diff', 'inter', 'ratio', 'pred'))

with open(traj_path, "r") as fin:
    for line in fin:
        line = line.split(' ')
        trajectory_index = line[1]
        cell_dict[trajectory_index] = list(map(int, line[2:]))
        true_label_dict[trajectory_index] = int(line[0])

# 计算异常得分
for k, v in cell_dict.items():
    # num = len(cell_dict)
    key_list = []
    for key in cell_dict.keys():
        key_list.append(key)
    # key_list 存储每条轨迹的index
    key_list.remove(k)
    sample = {}
    sample_keys = key_list
    sample_num = len(sample_keys)
    for k2 in sample_keys:
        sample.setdefault(k2, cell_dict[k2])

    diff_list = []  # |A-B|-|B-A|
    inter_list = []  # |A^B|
    for k1, v1 in sample.items():
        # v有v1没有
        diff = list(set(v).difference(set(v1)))
        # v1有 v没有
        diff1 = list(set(v1).difference(set(v)))
        # 两者共有
        inter = list(set(v).intersection(set(v1)))
        diff_list.append(len(diff) - len(diff1))
        inter_list.append(len(inter))

    index = k
    diff_mean = round(sum(diff_list) / sample_num, 4)
    inter_mean = round(sum(inter_list) / sample_num, 4)
    if int(inter_mean) == 0:
        ratio = 10000
    else:
        ratio = round(diff_mean / inter_mean, 4)
    if ratio > theta[0]:
        pred1_label_dict[k] = 1
        pred = 1
    elif (ratio > theta[1]) & (ratio <= theta[0]):
        pred1_label_dict[k] = 1
        pred = 1
    elif (ratio <= theta[1]) & (ratio >= theta[2]):
        pred1_label_dict[k] = 0
        pred = 0
    elif (ratio < theta[2]) & (ratio >= theta[3]):
        pred1_label_dict[k] = 1
        pred = 1
    else:
        pred1_label_dict[k] = 1
        pred = 1

    iso_df = iso_df.append({'index': index, 'diff': diff_mean, 'inter': inter_mean, 'ratio': ratio, 'pred': pred},
                           ignore_index=True)

# 计算指标
true_array = []
pred1_array = []
for trajectory_index in pred1_label_dict:
    true_array.append(true_label_dict[trajectory_index])
    pred1_array.append(pred1_label_dict[trajectory_index])

tp = [pred1_array[i] == 1 and true_array[i] == 1 for i in range(len(true_array))].count(True)
fn = [pred1_array[i] == 0 and true_array[i] == 1 for i in range(len(true_array))].count(True)
fp = [pred1_array[i] == 1 and true_array[i] == 0 for i in range(len(true_array))].count(True)
tn = [pred1_array[i] == 0 and true_array[i] == 0 for i in range(len(true_array))].count(True)
p = 0 if (tp + fp) == 0 else tp / (tp + fp)
r = 0 if (tp + fn) == 0 else tp / (tp + fn)
f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
print('p={}, r={}, f1={} '.format(p, r, f1))

# d第二轮
norm_df = iso_df[(iso_df['ratio'] >= phi[0]) & (iso_df['ratio'] <= phi[1])]
norm_index = norm_df['index'].values
for k, v in cell_dict.items():
    if k in norm_index:
        continue  # print('yes')
    else:
        sample = {}
        # sample_num = len(norm_index)
        for k2 in norm_index:
            sample.setdefault(k2, cell_dict[k2])

        diff_list = []  # |A-B|-|B-A|
        inter_list = []  # |A^B|
        for k1, v1 in sample.items():
            diff = list(set(v).difference(set(v1)))
            diff1 = list(set(v1).difference(set(v)))
            inter = list(set(v).intersection(set(v1)))
            diff_list.append(len(diff) - len(diff1))
            inter_list.append(len(inter))

        index = k
        val_dict = {'diff': diff_list, 'inter': inter_list}
        val_df = pd.DataFrame(val_dict)
        val_df.sort_values(by='inter', ascending=False, inplace=True)
        if kn >= val_df['inter'].count():
            kn = val_df['inter'].count()
        top_k = val_df.head(kn)
        diff_mean = round(sum(list(top_k['diff'])) / kn, 4)
        inter_mean = round(sum(list(top_k['inter'])) / kn, 4)

        if math.fabs(inter_mean) <= 0.0001:
            ratio = 10000
        else:
            ratio = round(diff_mean / inter_mean, 4)

        # trajectory classification
        if ratio > theta[0]:
            pred2_label_dict[k] = 1
            pred = 0
        elif (ratio > theta[1]) & (ratio <= theta[0]):
            pred2_label_dict[k] = 1
            pred = 1
        elif (ratio <= theta[1]) & (ratio >= theta[2]):
            pred2_label_dict[k] = 0
            pred = 2
        elif (ratio < theta[2]) & (ratio >= theta[3]):
            pred2_label_dict[k] = 1
            pred = 3
        else:
            pred2_label_dict[k] = 1
            pred = 4

        iso_df2 = iso_df2.append(
            {'index': index, 'diff': diff_mean, 'inter': inter_mean, 'ratio': ratio, 'pred': pred},
            ignore_index=True)

# 计算指标
pred2_array = []
true_array = []
for trajectory_index in pred2_label_dict:
    true_array.append(true_label_dict[trajectory_index])
    pred2_array.append(pred2_label_dict[trajectory_index])
print(pred2_array)
tp = [pred2_array[i] == 1 and true_array[i] == 1 for i in range(len(true_array))].count(True)
fn = [pred2_array[i] == 0 and true_array[i] == 1 for i in range(len(true_array))].count(True)
fp = [pred2_array[i] == 1 and true_array[i] == 0 for i in range(len(true_array))].count(True)
tn = [pred2_array[i] == 0 and true_array[i] == 0 for i in range(len(true_array))].count(True)
p = 0 if (tp + fp) == 0 else tp / (tp + fp)
r = 0 if (tp + fn) == 0 else tp / (tp + fn)
f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
print('p={}, r={}, f1={} '.format(p, r, f1))
