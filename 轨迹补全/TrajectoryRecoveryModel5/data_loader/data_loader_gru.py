import torch
import numpy as np
import pandas as pd
import copy
from torch.utils.data import Dataset
import os
from math import *
import matplotlib.pyplot as plt
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'


class tdrive_data_loader(Dataset):
    def __init__(self, path, mask, time_step):
        self.mask = mask
        self.path = path
        self.time_step = time_step
        self.__read_data__(mask)
        print('数据预处理结束')

    def __read_data__(self, mask_num):
        df = pd.read_csv(self.path, parse_dates=['time'], infer_datetime_format=True)
        # df['time'] = df['time'].apply(lambda x: pd.to_datetime(x))
        # 获取所有用户
        users_id = np.unique(df['userID'].values.reshape(-1))

        # 获取网格数量
        self.grids = df['number_grid'].values.reshape(-1)
        print('有{}个用户'.format(len(users_id)))
        print('最大网格单元编码{},最小网格单元编号{}'.format(max(self.grids), min(self.grids)))
        self.grid_count = max(self.grids) + 1
        self.user_count = max(users_id) + 1
        # 缺失的位置用0代替
        self.grids[np.isnan(self.grids)] = 0
        self.grid_unique = np.unique(self.grids)

        # 聚合历史轨迹序列
        self.aggregator_hist_traj_grid = []
        # 历史轨迹序列
        self.hist_traj_grid = []
        # 用户编码
        self.userId = []
        # 第几个位置被隐藏了
        self.mask_pos = []
        # 隐藏的位置值是多少
        self.mask_value = []
        # 隐藏的位置的经度
        self.mask_value_lat = []
        # 隐藏的位置的纬度
        self.mask_value_lng = []
        # 当前轨迹
        self.curr_traj_grid = []
        # 当前被隐藏掉一部分的的轨迹
        self.mask_curr_traj_grid = []

        # 根据用户选取数据
        for index, item in enumerate(users_id):
            # 用户编码
            self.userId.append(item)
            # 选取当前用户的轨迹
            current_df = df[df['userID'] == item]

            time_start = '2008-02-03 00:00:00'

            time_end = '2008-02-03 23:59:00'
            time_range = time_range = [[pd.to_datetime(time_start) + pd.Timedelta('{} Day'.format(index)),
                                        pd.to_datetime(time_end) + pd.Timedelta('{} Day'.format(index))] for
                                       index in range(4)]

            # 生成历史轨迹特征
            temp_hist_grid = []
            for item in time_range:
                current_grid = current_df[(current_df['time'] >= item[0]) & (current_df['time'] <= item[1])][
                    'number_grid'].values.reshape(-1).tolist()

                temp_hist_grid.append(current_grid)
            self.hist_traj_grid.append(copy.deepcopy(temp_hist_grid))

            # 聚合历史轨迹
            temp_aggregator_hist_traj_grid = []
            temp_hist_grid = np.array(temp_hist_grid)
            for i in range(len(temp_hist_grid[0])):
                temp = temp_hist_grid[:, i].tolist()
                counters = [temp.count(item) for item in temp_hist_grid[:, i]]
                temp_aggregator_hist_traj_grid.append(temp[counters.index(max(counters))])
            self.aggregator_hist_traj_grid.append(temp_aggregator_hist_traj_grid)

            # 生成当前轨迹特征  随机用1去替换5个位置的坐标 保留去除的位置索引，
            current_time_start = pd.to_datetime('2008-02-07 00:00:00')
            current_time_end = pd.to_datetime('2008-02-07 23:59:00')
            current_grid = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'number_grid'].values.reshape(-1).tolist()
            curr_lats = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lat'].values.reshape(-1).tolist()
            curr_lngs = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lng'].values.reshape(-1).tolist()

            self.curr_traj_grid.append(copy.deepcopy(current_grid))
            temp_mask_pos = []
            temp_mask_value = []
            temp_mask_lat = []
            temp_mask_lng = []
            temp_mask_curr_traj_grid = []

            while len(temp_mask_pos) < mask_num:
                pos = np.random.randint(0, self.time_step)

                if current_grid[pos] == 0 or pos in temp_mask_pos:
                    pass
                else:
                    temp_mask_pos.append(pos)
                    temp_mask_value.append(current_grid[pos])
                    temp_mask_lat.append(curr_lats[pos])
                    temp_mask_lng.append(curr_lngs[pos])
                    current_grid[pos] = 1
            self.mask_pos.append(temp_mask_pos)
            self.mask_value.append(temp_mask_value)
            self.mask_curr_traj_grid.append(current_grid)
            self.mask_value_lat.append(temp_mask_lat)
            self.mask_value_lng.append(temp_mask_lng)

        print(np.array(self.hist_traj_grid).shape, np.array(self.userId).shape, np.array(self.mask_pos).shape,
              np.array(self.mask_value).shape, np.array(self.curr_traj_grid).shape,
              np.array(self.mask_curr_traj_grid).shape, np.array(self.aggregator_hist_traj_grid).shape)
        self.hist_traj_grid = torch.Tensor(self.hist_traj_grid)
        self.userId = torch.Tensor(self.userId)
        self.mask_pos = torch.Tensor(self.mask_pos)
        self.mask_value = torch.Tensor(self.mask_value)
        self.mask_value_lat = torch.Tensor(self.mask_value_lat)
        self.mask_value_lng = torch.Tensor(self.mask_value_lng)
        self.curr_traj_grid = torch.Tensor(self.curr_traj_grid)
        self.mask_curr_traj_grid = torch.Tensor(self.mask_curr_traj_grid)
        self.aggregator_hist_traj_grid = torch.Tensor(self.aggregator_hist_traj_grid)

    def __len__(self):
        return len(self.hist_traj_grid)

    def __getitem__(self, item):
        # 历史轨迹数据 用户号 位置掩码 当前掩码轨迹 y(真实值) 当前完整轨迹(没啥用 预留)
        return self.hist_traj_grid[item], self.userId[item], self.mask_pos[item], self.mask_curr_traj_grid[item], \
               self.mask_value[item], self.curr_traj_grid[item], self.aggregator_hist_traj_grid[item], \
               self.mask_value_lat[item], self.mask_value_lng[item]


class shenzhen_data_loader(Dataset):
    def __init__(self, path, mask, time_step):
        self.mask = mask
        self.path = path
        self.time_step = time_step
        self.__read_data__(mask)
        print('数据预处理结束')

    def __read_data__(self, mask_num):
        df = pd.read_csv(self.path, parse_dates=['time'], infer_datetime_format=True)
        # df['time'] = df['time'].apply(lambda x: pd.to_datetime(x))

        # 获取所有用户
        users_id = np.unique(df['userID'].values.reshape(-1))

        # 获取网格数量
        self.grids = df['number_grid'].values.reshape(-1)
        print('有{}个用户'.format(len(users_id)))
        print('最大网格单元编码{},最小网格单元编号{}'.format(max(self.grids), min(self.grids)))
        self.grid_count = max(self.grids) + 1
        self.user_count = len(users_id)
        # 缺失的位置用0代替
        self.grids[np.isnan(self.grids)] = 0
        self.grid_unique = np.unique(self.grids)

        # 聚合历史轨迹序列
        self.aggregator_hist_traj_grid = []
        # 历史轨迹序列
        self.hist_traj_grid = []
        # 用户编码
        self.userId = []
        # 第几个位置被隐藏了
        self.mask_pos = []
        # 隐藏的位置值是多少
        self.mask_value = []
        # 隐藏的位置的经度
        self.mask_value_lat = []
        # 隐藏的位置的纬度
        self.mask_value_lng = []
        # 当前轨迹
        self.curr_traj_grid = []
        # 当前被隐藏掉一部分的的轨迹
        self.mask_curr_traj_grid = []

        # 根据用户选取数据
        for index, user in enumerate(users_id):
            # 用户编码
            self.userId.append(user)
            # 选取当前用户的轨迹
            current_df = df[df['userID'] == user]

            time_start = '2009-09-04 00:00:00'

            time_end = '2009-09-04 23:59:59'
            time_range = time_range = [[pd.to_datetime(time_start) + pd.Timedelta('{} Day'.format(index)),
                                        pd.to_datetime(time_end) + pd.Timedelta('{} Day'.format(index))] for
                                       index in range(4)]

            # 生成历史轨迹特征
            temp_hist_grid = []
            for item in time_range:
                current_grid = current_df[(current_df['time'] >= item[0]) & (current_df['time'] <= item[1])][
                    'number_grid'].values.reshape(-1).tolist()

                temp_hist_grid.append(current_grid[:self.time_step])

            self.hist_traj_grid.append(temp_hist_grid)

            # 聚合历史轨迹
            temp_aggregator_hist_traj_grid = []
            temp_hist_grid = np.array(temp_hist_grid)

            for i in range(len(temp_hist_grid[0])):
                temp = temp_hist_grid[:, i].tolist()
                counters = [temp.count(item) for item in temp_hist_grid[:, i]]
                temp_aggregator_hist_traj_grid.append(temp[counters.index(max(counters))])
            self.aggregator_hist_traj_grid.append(temp_aggregator_hist_traj_grid)

            # 生成当前轨迹特征  随机用1去替换5个位置的坐标 保留去除的位置索引，
            current_time_start = pd.to_datetime('2009-09-08 00:00:00')
            current_time_end = pd.to_datetime('2009-09-08 23:59:59')
            current_grid = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'number_grid'].values.reshape(-1).tolist()
            curr_lats = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lat'].values.reshape(-1).tolist()
            curr_lngs = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lng'].values.reshape(-1).tolist()

            self.curr_traj_grid.append(copy.deepcopy(current_grid))
            temp_mask_pos = []
            temp_mask_value = []
            temp_mask_lat = []
            temp_mask_lng = []
            temp_mask_curr_traj_grid = []

            while len(temp_mask_pos) < mask_num:
                pos = np.random.randint(0, self.time_step)
                # current_grid[pos] == 0
                if current_grid[pos] == 0 or pos in temp_mask_pos:
                    pass
                else:
                    temp_mask_pos.append(pos)
                    temp_mask_value.append(current_grid[pos])
                    temp_mask_lat.append(curr_lats[pos])
                    temp_mask_lng.append(curr_lngs[pos])
                    current_grid[pos] = 1

            self.mask_pos.append(temp_mask_pos)
            self.mask_value.append(temp_mask_value)
            self.mask_curr_traj_grid.append(current_grid)
            self.mask_value_lat.append(temp_mask_lat)
            self.mask_value_lng.append(temp_mask_lng)

        print(np.array(self.hist_traj_grid).shape, np.array(self.userId).shape, np.array(self.mask_pos).shape,
              np.array(self.mask_value).shape, np.array(self.curr_traj_grid).shape,
              np.array(self.mask_curr_traj_grid).shape)
        self.hist_traj_grid = torch.Tensor(self.hist_traj_grid)
        self.userId = torch.Tensor([item for item in range(len(self.userId))])
        self.mask_pos = torch.Tensor(self.mask_pos)
        self.mask_value = torch.Tensor(self.mask_value)
        self.mask_value_lat = torch.Tensor(self.mask_value_lat)
        self.mask_value_lng = torch.Tensor(self.mask_value_lng)
        self.curr_traj_grid = torch.Tensor(self.curr_traj_grid)
        self.mask_curr_traj_grid = torch.Tensor(self.mask_curr_traj_grid)
        self.aggregator_hist_traj_grid = torch.Tensor(self.aggregator_hist_traj_grid)

    def __len__(self):
        return len(self.hist_traj_grid)

    def __getitem__(self, item):
        # 历史轨迹数据 用户号 位置掩码 当前掩码轨迹 y(真实值) 当前完整轨迹(没啥用 预留)
        return self.hist_traj_grid[item], self.userId[item], self.mask_pos[item], self.mask_curr_traj_grid[item], \
               self.mask_value[item], self.curr_traj_grid[item], self.aggregator_hist_traj_grid[item], \
               self.mask_value_lat[item], self.mask_value_lng[item]


class chengdu_data_loader(Dataset):
    def __init__(self, path, mask, time_step):
        self.mask = mask
        self.path = path
        self.time_step = time_step
        self.__read_data__(mask)
        print('数据预处理结束')

    def __read_data__(self, mask_num):
        df = pd.read_csv(self.path, parse_dates=['time'], infer_datetime_format=True)
        # df['time'] = df['time'].apply(lambda x: pd.to_datetime(x))

        # 获取所有用户
        users_id = np.unique(df['userID'].values.reshape(-1))

        # 获取网格数量
        self.grids = df['number_grid'].values.reshape(-1)

        print('有{}个用户'.format(len(users_id)))
        print('最大网格单元编码{},最小网格单元编号{}'.format(np.nanmax(self.grids), np.nanmin(self.grids)))
        self.grid_count = np.nanmax(self.grids) + 1
        self.user_count = len(users_id)
        # 缺失的位置用0代替
        self.grids[np.isnan(self.grids)] = 0
        self.grid_unique = np.unique(self.grids)

        # 聚合历史轨迹序列
        self.aggregator_hist_traj_grid = []
        # 历史轨迹序列
        self.hist_traj_grid = []
        # 用户编码
        self.userId = []
        # 第几个位置被隐藏了
        self.mask_pos = []
        # 隐藏的位置值是多少
        self.mask_value = []
        # 隐藏的位置的经度
        self.mask_value_lat = []
        # 隐藏的位置的纬度
        self.mask_value_lng = []
        # 当前轨迹
        self.curr_traj_grid = []
        # 当前被隐藏掉一部分的的轨迹
        self.mask_curr_traj_grid = []

        # 根据用户选取数据
        for index, user in enumerate(users_id):
            # 用户编码
            self.userId.append(user)
            # 选取当前用户的轨迹
            current_df = df[df['userID'] == user]

            time_start = '2014-08-08 00:00:00'

            time_end = '2014-08-08 23:59:59'
            time_range = [[pd.to_datetime(time_start) + pd.Timedelta('{} Day'.format(index)),
                           pd.to_datetime(time_end) + pd.Timedelta('{} Day'.format(index))] for
                          index in range(4)]

            # 生成历史轨迹特征
            temp_hist_grid = []
            for item in time_range:
                current_grid = current_df[(current_df['time'] >= item[0]) & (current_df['time'] <= item[1])][
                    'number_grid'].values.reshape(-1).tolist()

                temp_hist_grid.append(current_grid[:self.time_step])

            self.hist_traj_grid.append(temp_hist_grid)

            # 聚合历史轨迹
            temp_aggregator_hist_traj_grid = []
            temp_hist_grid = np.array(temp_hist_grid)

            for i in range(len(temp_hist_grid[0])):
                temp = temp_hist_grid[:, i].tolist()
                counters = [temp.count(item) for item in temp_hist_grid[:, i]]
                temp_aggregator_hist_traj_grid.append(temp[counters.index(max(counters))])
            self.aggregator_hist_traj_grid.append(temp_aggregator_hist_traj_grid)

            # 生成当前轨迹特征  随机用1去替换5个位置的坐标 保留去除的位置索引，
            current_time_start = pd.to_datetime('2014-08-12 00:00:00')
            current_time_end = pd.to_datetime('2014-08-12 23:59:59')
            current_grid = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'number_grid'].values.reshape(-1).tolist()
            curr_lats = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lat'].values.reshape(-1).tolist()
            curr_lngs = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lng'].values.reshape(-1).tolist()

            self.curr_traj_grid.append(copy.deepcopy(current_grid))
            temp_mask_pos = []
            temp_mask_value = []
            temp_mask_lat = []
            temp_mask_lng = []
            temp_mask_curr_traj_grid = []

            while len(temp_mask_pos) < mask_num:
                pos = np.random.randint(0, self.time_step)
                # current_grid[pos] == 0
                if current_grid[pos] == 0 or pos in temp_mask_pos:
                    pass
                else:
                    temp_mask_pos.append(pos)
                    temp_mask_value.append(current_grid[pos])
                    temp_mask_lat.append(curr_lats[pos])
                    temp_mask_lng.append(curr_lngs[pos])
                    current_grid[pos] = 1

            self.mask_pos.append(temp_mask_pos)
            self.mask_value.append(temp_mask_value)
            self.mask_curr_traj_grid.append(current_grid)
            self.mask_value_lat.append(temp_mask_lat)
            self.mask_value_lng.append(temp_mask_lng)

        print(np.array(self.hist_traj_grid).shape, np.array(self.userId).shape, np.array(self.mask_pos).shape,
              np.array(self.mask_value).shape, np.array(self.curr_traj_grid).shape,
              np.array(self.mask_curr_traj_grid).shape)
        self.hist_traj_grid = torch.Tensor(self.hist_traj_grid)
        self.userId = torch.Tensor([item for item in range(len(self.userId))])
        self.mask_pos = torch.Tensor(self.mask_pos)
        self.mask_value = torch.Tensor(self.mask_value)
        self.mask_value_lat = torch.Tensor(self.mask_value_lat)
        self.mask_value_lng = torch.Tensor(self.mask_value_lng)
        self.curr_traj_grid = torch.Tensor(self.curr_traj_grid)
        self.mask_curr_traj_grid = torch.Tensor(self.mask_curr_traj_grid)
        self.aggregator_hist_traj_grid = torch.Tensor(self.aggregator_hist_traj_grid)

    def __len__(self):
        return len(self.hist_traj_grid)

    def __getitem__(self, item):
        # 历史轨迹数据 用户号 位置掩码 当前掩码轨迹 y(真实值) 当前完整轨迹(没啥用 预留)
        return self.hist_traj_grid[item], self.userId[item], self.mask_pos[item], self.mask_curr_traj_grid[item], \
               self.mask_value[item], self.curr_traj_grid[item], self.aggregator_hist_traj_grid[item], \
               self.mask_value_lat[item], self.mask_value_lng[item]
