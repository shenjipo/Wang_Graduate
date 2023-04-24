import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data_loader.data_loader_periodicmove import chengdu_data_loader
from model.periodicmove_model import PeriodiceMove
import torch.nn as nn
from torch import optim
import datetime
from data_loader.data_loader_periodicmove import construct_graph
from utils import calDisByGrid, calDisBylatlng

np.random.seed(5)
mask = 10
time_step = 144
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# t_drive_dispersed_500_116.00211_116.79979_39.5018_40.19988
# shenzhen_taxi/shenzhen_texi_10_500_113.6139_114.5961_22.461_22.9998.csv
t = chengdu_data_loader(path='processed_data/chengdu_taxi/chengdu_interval_10_500_103.2697_104.6097_30.2907_31.0325(1).csv',
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
d_model = 144
heads = 4
dropout = 0.5
step = 1
'''
设置召回指标
'''
recall = 10

train_set, val_set, test_set = torch.utils.data.random_split(t, [train_size, val_size, test_size])
data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_set, batch_size=512, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(test_set, batch_size=512, shuffle=True, num_workers=0, drop_last=False)

model = PeriodiceMove(d_model=d_model, heads=heads, grid_count=t.grid_count, user_count=t.user_count,
                      time_step=time_step, step=step)

# 3.训练模型
train_epochs = 2000
model_optim = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.00)
schedule = optim.lr_scheduler.StepLR(model_optim, step_size=400, gamma=0.5)
criterion = nn.NLLLoss()
model.to(device)
max_val = 0
for epoch in range(train_epochs):
    iter_count = 0
    train_loss = []
    model.train()

    for i, (hist_traj_grid, userId, mask_pos, mask_curr_traj_grid, mask_value, curr_traj_grid,
            aggregator_hist_traj_grid, _, __) in enumerate(data_loader):
        # 梯度清零
        model_optim.zero_grad()

        hist_traj_in_grid, hist_traj_A, hist_traj_items = construct_graph(
            hist_traj_grid.reshape(-1, time_step).detach().numpy())
        curr_traj_in_grid, curr_traj_A, curr_traj_items = construct_graph(
            mask_curr_traj_grid.reshape(-1, time_step).detach().numpy())

        hist_traj_in_grid = torch.Tensor(hist_traj_in_grid).to(torch.int64).to(device)
        hist_traj_A = torch.Tensor(hist_traj_A).to(torch.float).to(device)
        hist_traj_items = torch.Tensor(hist_traj_items).to(torch.int64).to(device)
        curr_traj_in_grid = torch.Tensor(curr_traj_in_grid).to(torch.int64).to(device)
        curr_traj_A = torch.Tensor(curr_traj_A).to(torch.float).to(device)
        curr_traj_items = torch.Tensor(curr_traj_items).to(torch.int64).to(device)

        hist_traj_grid = hist_traj_grid.to(torch.int64).to(device)
        userId = userId.to(torch.int64).to(device)
        mask_pos = mask_pos.to(torch.int64).to(device)
        mask_curr_traj_grid = mask_curr_traj_grid.to(torch.int64).to(device)
        mask_value = mask_value.to(torch.int64).to(device)

        outputs = model(hist_traj_grid, userId, mask_pos, mask_curr_traj_grid, hist_traj_A, curr_traj_A,
                        hist_traj_items, curr_traj_items, hist_traj_in_grid, curr_traj_in_grid)

        # 计算损失

        mask_value = mask_value.reshape(-1)

        loss = criterion(outputs, mask_value - 2)

        train_loss.append(loss.item())

        # 梯度反向传播
        loss.backward()

        model_optim.step()
    with torch.no_grad():
        for j, (val_hist_traj_grid, val_userId, val_mask_pos, val_mask_curr_traj_grid, val_mask_value,
                val_curr_traj_grid, val_aggregator_hist_traj_grid, _, __) in enumerate(val_loader):

            val_hist_traj_in_grid, val_hist_traj_A, val_hist_traj_items = construct_graph(
                val_hist_traj_grid.reshape(-1, time_step).detach().numpy())
            val_curr_traj_in_grid, val_curr_traj_A, val_curr_traj_items = construct_graph(
                val_mask_curr_traj_grid.reshape(-1, time_step).detach().numpy())

            val_hist_traj_in_grid = torch.Tensor(val_hist_traj_in_grid).to(torch.int64).to(device)
            val_hist_traj_A = torch.Tensor(val_hist_traj_A).to(torch.float).to(device)
            val_hist_traj_items = torch.Tensor(val_hist_traj_items).to(torch.int64).to(device)
            val_curr_traj_in_grid = torch.Tensor(val_curr_traj_in_grid).to(torch.int64).to(device)
            val_curr_traj_A = torch.Tensor(val_curr_traj_A).to(torch.float).to(device)
            val_curr_traj_items = torch.Tensor(val_curr_traj_items).to(torch.int64).to(device)

            val_hist_traj_grid = val_hist_traj_grid.to(torch.int64).to(device)
            val_userId = val_userId.to(torch.int64).to(device)
            val_mask_pos = val_mask_pos.to(torch.int64).to(device)
            val_mask_curr_traj_grid = val_mask_curr_traj_grid.to(torch.int64).to(device)
            val_mask_value = val_mask_value.to(torch.int64).to(device)

            # 预测
            test_outputs = model(val_hist_traj_grid, val_userId, val_mask_pos, val_mask_curr_traj_grid, val_hist_traj_A,
                                 val_curr_traj_A,
                                 val_hist_traj_items, val_curr_traj_items, val_hist_traj_in_grid, val_curr_traj_in_grid)

            val_mask_value = val_mask_value.reshape(-1)

            value, index = torch.topk(test_outputs, recall, dim=1)
            # value, index = outputs.max(1)
            index = index.cpu().detach().numpy().reshape(-1, recall)
            true_index = (val_mask_value - 2).cpu().detach().numpy().reshape(-1)
            true_count = 0
            fasle_count = 0
            for k in range(len(index)):
                if true_index[k] in index[k]:
                    true_count += 1
                else:
                    fasle_count += 1
            if round(true_count / (fasle_count + true_count), 4) > max_val:
                # 4.保存模型
                torch.save(model.state_dict(),
                           'pars/periodicmove/periodicmove_chengdu_recall@{}_{}'.format(recall,
                                                                                         datetime.datetime.now().strftime(
                                                                                             '%Y-%m-%d %H_%M_%S')))
                max_val = round(true_count / (fasle_count + true_count), 4)
                print('验证集:正确的位置个数{},错误的位置个数{},正确率{}'.format(true_count, fasle_count,
                                                             round(
                                                                 true_count / (fasle_count + true_count),
                                                                 4)))

        for j, (test_hist_traj_grid, test_userId, test_mask_pos, test_mask_curr_traj_grid, test_mask_value,
                test_curr_traj_grid, aggregator_hist_traj_grid, mask_value_lat, mask_value_lng) in enumerate(
            test_loader):

            test_hist_traj_in_grid, test_hist_traj_A, test_hist_traj_items = construct_graph(
                test_hist_traj_grid.reshape(-1, time_step).detach().numpy())
            test_curr_traj_in_grid, test_curr_traj_A, test_curr_traj_items = construct_graph(
                test_mask_curr_traj_grid.reshape(-1, time_step).detach().numpy())

            test_hist_traj_in_grid = torch.Tensor(test_hist_traj_in_grid).to(torch.int64).to(device)
            test_hist_traj_A = torch.Tensor(test_hist_traj_A).to(torch.float).to(device)
            test_hist_traj_items = torch.Tensor(test_hist_traj_items).to(torch.int64).to(device)
            test_curr_traj_in_grid = torch.Tensor(test_curr_traj_in_grid).to(torch.int64).to(device)
            test_curr_traj_A = torch.Tensor(test_curr_traj_A).to(torch.float).to(device)
            test_curr_traj_items = torch.Tensor(test_curr_traj_items).to(torch.int64).to(device)

            test_hist_traj_grid = test_hist_traj_grid.to(torch.int64).to(device)
            test_userId = test_userId.to(torch.int64).to(device)
            test_mask_pos = test_mask_pos.to(torch.int64).to(device)
            test_mask_curr_traj_grid = test_mask_curr_traj_grid.to(torch.int64).to(device)
            test_mask_value = test_mask_value.to(torch.int64).to(device)
            mask_value_lat = mask_value_lat.to(torch.float64).to(device)
            mask_value_lng = mask_value_lng.to(torch.float64).to(device)

            # 预测
            test_outputs = model(test_hist_traj_grid, test_userId, test_mask_pos, test_mask_curr_traj_grid,
                                 test_hist_traj_A, test_curr_traj_A,
                                 test_hist_traj_items, test_curr_traj_items, test_hist_traj_in_grid,
                                 test_curr_traj_in_grid)

            test_mask_value = test_mask_value.reshape(-1)

            value, index = torch.topk(test_outputs, recall, dim=1)
            # value, index = outputs.max(1)
            index = index.cpu().detach().numpy().reshape(-1, recall)
            true_index = (test_mask_value - 2).cpu().detach().numpy().reshape(-1)
            true_lats = mask_value_lat.cpu().detach().numpy().reshape(-1)
            true_lngs = mask_value_lng.cpu().detach().numpy().reshape(-1)

            true_count = 0
            fasle_count = 0
            testDisError = 0
            repairDisError = 0
            for k in range(len(index)):
                testDisError += calDisBylatlng(true_index[k], index[k][0], true_lats[k], true_lngs[k], 'chengdu')
                if true_index[k] in index[k]:
                    # repairDisError += calDisBylatlng(true_index[k], index[k][0], true_lats[k], true_lngs[k], 'chengdu')
                    true_count += 1
                else:
                    fasle_count += 1
            if true_count != 0:
                print('测试集:正确的位置个数{},错误的位置个数{},正确率{},平均误差1--{},平均误差2--{}'.
                      format(true_count, fasle_count, round(true_count / (fasle_count + true_count), 4),
                             round(testDisError / len(index), 4), round(repairDisError / true_count, 4)))
    schedule.step()
    print(
        '当前第{}轮训练结束，共{}轮，进行了{}%,当前损失{},当前学习率{}'.format(epoch, train_epochs, epoch / train_epochs * 100,
                                                       np.average(train_loss), schedule.get_lr()))
