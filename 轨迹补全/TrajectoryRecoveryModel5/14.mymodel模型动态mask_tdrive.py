from utils import calDisByGrid, calDisBylatlng
import my_model_test
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data_loader.data_loader_mymodel import tdrive_data_loader
from model.mymodel import MyModel
import torch.nn as nn
from torch import optim
import datetime
from data_loader.data_loader_mymodel import construct_graph


mask = 10
time_step = 144
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
recall = 1
dm = 5

val_loader = None
test_loader = None
grid_count = None
user_count = None
train_loader = {}
for i in range(4,4+dm):
    np.random.seed(i + 1)
    temp_set = tdrive_data_loader(
        path='processed_data/t_drive/t_drive_dispersed_500_116.00211_116.79979_39.5018_40.19988.csv', mask=mask,
        time_step=time_step)
    train_size = int(len(temp_set) * 0.7)
    val_size = int(len(temp_set) * 0.1)
    test_size = len(temp_set) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(temp_set, [train_size, val_size, test_size])
    temp_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    grid_count = temp_set.grid_count
    user_count = temp_set.user_count
    train_loader[i-4] = temp_train_loader
    if i == 4:
        val_loader = DataLoader(val_set, batch_size=batch_size * 3, shuffle=True, num_workers=0, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=batch_size * 3, shuffle=True, num_workers=0, drop_last=False)

print('动态mask结束!!!')
model = MyModel(d_model=d_model, heads=heads, grid_count=grid_count, user_count=user_count, time_step=time_step,
                step=step)

# 3.训练模型
train_epochs = 2000

model_optim = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.005)
schedule = optim.lr_scheduler.StepLR(model_optim, step_size=400, gamma=0.5)
criterion = nn.NLLLoss()
model.to(device)
max_val = 0
for epoch in range(train_epochs):
    iter_count = 0
    train_loss = []
    model.train()

    for i, (
    hist_traj_grid, userId, mask_pos, mask_curr_traj_grid, mask_value, curr_traj_grid, dtw_border, _, __) in enumerate(
            train_loader[epoch % dm]):
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

        mask_pos = mask_pos.to(torch.int64).to(device)
        mask_curr_traj_grid = mask_curr_traj_grid.to(torch.int64).to(device)
        mask_value = mask_value.to(torch.int64).to(device)
        dtw_border = dtw_border.to(torch.int64).to(device)

        outputs = model(hist_traj_grid, dtw_border, mask_pos, mask_curr_traj_grid, hist_traj_A, curr_traj_A,
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
                val_curr_traj_grid, val_dtw_border, _, __) in enumerate(val_loader):

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
            val_mask_pos = val_mask_pos.to(torch.int64).to(device)
            val_mask_curr_traj_grid = val_mask_curr_traj_grid.to(torch.int64).to(device)
            val_mask_value = val_mask_value.to(torch.int64).to(device)
            val_dtw_border = val_dtw_border.to(torch.int64).to(device)

            # 预测
            test_outputs = model(val_hist_traj_grid, val_dtw_border, val_mask_pos, val_mask_curr_traj_grid,
                                 val_hist_traj_A,
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
                           'pars/mymodel/mymodel_tdrive_recall@{}_{}_dm'.format(recall,
                                                                                   datetime.datetime.now().strftime(
                                                                                       '%Y-%m-%d %H_%M_%S')))
                max_val = round(true_count / (fasle_count + true_count), 4)
                print('验证集:正确的位置个数{},错误的位置个数{},正确率{}'.format(true_count, fasle_count,
                                                             round(
                                                                 true_count / (fasle_count + true_count),
                                                                 4)))

        for j, (test_hist_traj_grid, test_userId, test_mask_pos, test_mask_curr_traj_grid, test_mask_value,
                test_curr_traj_grid, test_dtw_border, mask_value_lat, mask_value_lng) in enumerate(test_loader):

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
            test_mask_pos = test_mask_pos.to(torch.int64).to(device)
            test_mask_curr_traj_grid = test_mask_curr_traj_grid.to(torch.int64).to(device)
            test_mask_value = test_mask_value.to(torch.int64).to(device)
            test_dtw_border = test_dtw_border.to(torch.int64).to(device)
            mask_value_lat = mask_value_lat.to(torch.float64).to(device)
            mask_value_lng = mask_value_lng.to(torch.float64).to(device)

            # 预测
            test_outputs = model(test_hist_traj_grid, test_dtw_border, test_mask_pos, test_mask_curr_traj_grid,
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
                testDisError += calDisBylatlng(true_index[k], index[k][0],true_lats[k], true_lngs[k], 'beijing')
                if true_index[k] in index[k]:
                    # repairDisError += calDisBylatlng(true_index[k], index[k][0], true_lats[k], true_lngs[k], 'beijing')
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
