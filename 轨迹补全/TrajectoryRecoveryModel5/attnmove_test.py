import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data_loader.data_loader_attnmove import tdrive_data_loader, shenzhen_data_loader, chengdu_data_loader
from model.attnmove_model import AttnMoveModel
import torch.nn as nn
from torch import optim
import datetime
from utils import calDisBylatlng

if __name__ == '__main__':

    # 显示全部数据
    np.set_printoptions(threshold=np.inf)
    np.random.seed(5)
    mask = 10
    time_step = 144
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # t_drive/t_drive_dispersed_500_116.00211_116.79979_39.5018_40.19988
    # baxi/baxi_bus_10_500_43.16_43.7226_22.7858_23.0668(3)
    # shenzhen_taxi/shenzhen_texi_10_500_113.6139_114.5961_22.461_22.9998
    # t = tdrive_data_loader(path='processed_data/t_drive/t_drive_dispersed_500_116.00211_116.79979_39.5018_40.19988.csv',
    #                        mask=mask, time_step=time_step)
    # t = shenzhen_data_loader(path='processed_data/shenzhen_taxi/shenzhen_texi_10_500_113.6139_114.5961_22.461_22.9998.csv',
    #                        mask=mask, time_step=time_step)
    t = chengdu_data_loader(
        path='processed_data/chengdu_taxi/chengdu_interval_10_500_103.2697_104.6097_30.2907_31.0325(1).csv',
        mask=mask, time_step=time_step)
    train_size = int(len(t) * 0.7)
    val_size = int(len(t) * 0.1)
    test_size = len(t) - train_size - val_size
    print('train_size={},test_size={}'.format(train_size, test_size))
    # 设置随机数种子
    torch.manual_seed(5)

    '''
    设置超参数
    '''
    batch_size = 128
    # 120 8
    # 144 4
    # 120 8
    d_model = 120
    heads = 8
    dropout = 0.5
    recall = 1

    train_set, val_set, test_set = torch.utils.data.random_split(t, [train_size, val_size, test_size])
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=True, num_workers=0, drop_last=False)

    model = AttnMoveModel(d_model=d_model, heads=heads, grid_count=t.grid_count, user_count=t.user_count,
                          dropout=dropout)

    # 3.加载模型
    model.load_state_dict(
        torch.load('pars/attnmove/attnmove_chengdu_recall@1_2022-06-04 08_41_14', map_location=torch.device('cpu')))
    max_val = 0
    with torch.no_grad():
        for j, (test_hist_traj_grid, test_userId, test_mask_pos, test_mask_curr_traj_grid, test_mask_value,
                test_curr_traj_grid, test_aggregator_hist_traj_grid, mask_value_lat, mask_value_lng) in enumerate(
            test_loader):
            test_hist_traj_grid = test_hist_traj_grid.to(torch.int64).to(device)
            test_userId = test_userId.to(torch.int64).to(device)
            test_mask_pos = test_mask_pos.to(torch.int64).to(device)
            test_mask_curr_traj_grid = test_mask_curr_traj_grid.to(torch.int64).to(device)
            test_mask_value = test_mask_value.to(torch.int64).to(device)
            mask_value_lat = mask_value_lat.to(torch.float64).to(device)
            mask_value_lng = mask_value_lng.to(torch.float64).to(device)
            test_aggregator_hist_traj_grid = test_aggregator_hist_traj_grid.to(torch.int64).to(device)
            # 预测
            test_outputs = model(test_hist_traj_grid, test_userId, test_mask_pos, test_mask_curr_traj_grid,
                                 test_aggregator_hist_traj_grid)

            test_mask_value = test_mask_value.reshape(-1)

            value, index = torch.topk(test_outputs, recall, dim=1)
            # value, index = outputs.max(1)
            index = index.cpu().detach().numpy().reshape(-1, recall)
            true_index = (test_mask_value - 2).cpu().detach().numpy().reshape(-1)
            true_lats = mask_value_lat.cpu().detach().numpy().reshape(-1)
            true_lngs = mask_value_lng.cpu().detach().numpy().reshape(-1)
            userId = test_userId.cpu().detach().numpy().reshape(-1)

            true_count = 0
            fasle_count = 0
            testDisError = 0
            repairDisError = 0
            recovery_df = pd.DataFrame(
                columns=(
                'userID', 'trueGrid', 'predGrid', 'trueLat', 'trueLng', 'predLat', 'predLng', 'disError', 'res'))
            for k in range(len(index)):
                temp_error, pred_lat, pred_lng, manhattan = calDisBylatlng(true_index[k], index[k][0], true_lats[k],
                                                                           true_lngs[k],
                                                                           'chengdu')
                testDisError += temp_error
                if true_index[k] in index[k]:
                    # repairDisError += calDisBylatlng(true_index[k], index[k][0], true_lats[k], true_lngs[k], 'baxi')
                    true_count += 1
                else:
                    fasle_count += 1
                recovery_df = recovery_df.append([{
                    'userID': userId[k // 10],
                    'trueGrid': true_index[k],
                    'predGrid': index[k][0],
                    'trueLat': true_lats[k],
                    'trueLng': true_lngs[k],
                    'predLat': pred_lat,
                    'predLng': pred_lng,
                    'disError': temp_error,
                    'res': 0 if true_index[k] in index[k] else 1,
                    'manhattan': manhattan
                }], ignore_index=True)

            recovery_df.to_csv('res/chengdu/attnmove_chengdu_error.csv', index=False)
            if true_count != 0:
                print('测试集:正确的位置个数{},错误的位置个数{},正确率{},平均误差1--{},平均误差2--{}'.
                      format(true_count, fasle_count, round(true_count / (fasle_count + true_count), 4),
                             round(testDisError / len(index), 4), round(repairDisError / true_count, 4)))
