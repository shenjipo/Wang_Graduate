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
        dtw_df = pd.read_csv(
            'processed_data/t_drive/t_drive_dispersed_500_116.00211_116.79979_39.5018_40.19988_dtw.csv')
        self.dtw_border = dtw_df['dtw'].values.reshape(-1).tolist()

        # df['time'] = df['time'].apply(lambda x: pd.to_datetime(x))
        # 获取所有用户
        users_id = np.unique(df['userID'].values.reshape(-1))

        # 获取网格数量
        self.grids = df['number_grid'].values.reshape(-1)
        print('有{}个用户'.format(len(users_id)))
        print('最大网格单元编码{},最小网格单元编号{}'.format(np.nanmax(self.grids), np.nanmin(self.grids)))

        self.user_count = len(users_id)
        # 缺失的位置用0代替
        self.grids[np.isnan(self.grids)] = 0
        self.grid_unique = np.unique(self.grids)
        self.grid_to_number = {}
        self.number_to_grid = {}
        self.grid_count = max(self.grid_unique) + 1
        self.grid_to_number[0] = 0
        self.number_to_grid[0] = 0
        self.grid_to_number[1] = 1
        self.number_to_grid[1] = 1
        for i in range(1, len(self.grid_unique)):
            self.grid_to_number[self.grid_unique[i]] = i + 1
            self.number_to_grid[i + 1] = self.grid_unique[i]

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
            self.userId.append(index)
            # 选取当前用户的轨迹
            current_df = df[df['userID'] == item]

            time_start = '2008-02-03 00:00:00'

            time_end = '2008-02-03 23:59:00'
            time_range = [[pd.to_datetime(time_start) + pd.Timedelta('{} Day'.format(index)),
                           pd.to_datetime(time_end) + pd.Timedelta('{} Day'.format(index))] for
                          index in range(4)]

            # 生成历史轨迹特征
            temp_hist_grid = []
            for item in time_range:
                current_grid = current_df[(current_df['time'] >= item[0]) & (current_df['time'] <= item[1])][
                    'number_grid'].values.reshape(-1).tolist()
                temp_hist_grid.append([item for item in current_grid])
                # temp_hist_grid.append([self.grid_to_number[item] for item in current_grid])
            self.hist_traj_grid.append(copy.deepcopy(temp_hist_grid))

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
            self.curr_traj_grid.append([item for item in copy.deepcopy(current_grid)])
            # self.curr_traj_grid.append([self.grid_to_number[item] for item in copy.deepcopy(current_grid)])
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
            # self.mask_value.append([self.grid_to_number[item] for item in temp_mask_value])
            # self.mask_curr_traj_grid.append([self.grid_to_number[item] for item in current_grid])
            self.mask_value.append([item for item in temp_mask_value])
            self.mask_curr_traj_grid.append([item for item in current_grid])
            self.mask_value_lat.append(temp_mask_lat)
            self.mask_value_lng.append(temp_mask_lng)

            # 聚合历史轨迹
            temp_aggregator_hist_traj_grid = []
            temp_hist_grid = np.array(temp_hist_grid)
            for i in range(len(temp_hist_grid[0])):
                temp = temp_hist_grid[:, i].tolist()
                counters = [temp.count(item) for item in temp_hist_grid[:, i]]
                temp_aggregator_hist_traj_grid.append(temp[counters.index(max(counters))])

            self.aggregator_hist_traj_grid.append(temp_aggregator_hist_traj_grid)

        print(np.array(self.hist_traj_grid).shape, np.array(self.userId).shape, np.array(self.mask_pos).shape,
              np.array(self.mask_value).shape, np.array(self.curr_traj_grid).shape, np.array(self.dtw_border).shape,
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
        self.dtw_border = torch.Tensor(self.dtw_border)

    def __len__(self):
        return len(self.hist_traj_grid)

    def __getitem__(self, item):
        # 历史轨迹数据 用户号 位置掩码 当前掩码轨迹 y(真实值) 当前完整轨迹(没啥用 预留)
        return self.hist_traj_grid[item], self.userId[item], self.mask_pos[item], self.mask_curr_traj_grid[item], \
               self.mask_value[item], self.curr_traj_grid[item], self.dtw_border[item], self.mask_value_lat[item], \
               self.mask_value_lng[item]


class baxi_data_loader(Dataset):
    def __init__(self, path, mask, time_step):
        self.mask = mask
        self.path = path
        self.time_step = time_step
        self.__read_data__(mask)
        print('数据预处理结束')

    def __read_data__(self, mask_num):
        df = pd.read_csv(self.path, parse_dates=['time'], infer_datetime_format=True)
        # baxi_bus_10_500_43.16_43.7226_22.7857_23.0668_dtw(1)
        # baxi_bus_10_500_43.16_43.7226_22.7858_23.0668_dtw(4)
        dtw_df = pd.read_csv('processed_data/baxi/baxi_bus_10_500_43.16_43.7226_22.7857_23.0668_dtw(1).csv')
        self.dtw_border = dtw_df['dtw'].values.reshape(-1).tolist()

        # df['time'] = df['time'].apply(lambda x: pd.to_datetime(x))
        # 获取所有用户
        users_id = np.unique(df['userID'].values.reshape(-1))

        # 获取网格数量
        self.grids = df['number_grid'].values.reshape(-1)
        print('有{}个用户'.format(len(users_id)))
        print('最大网格单元编码{},最小网格单元编号{}'.format(np.nanmax(self.grids), np.nanmin(self.grids)))

        self.user_count = len(users_id)
        # 缺失的位置用0代替
        self.grids[np.isnan(self.grids)] = 0
        self.grid_unique = np.unique(self.grids)
        self.grid_to_number = {}
        self.number_to_grid = {}
        self.grid_count = max(self.grid_unique) + 1
        self.grid_to_number[0] = 0
        self.number_to_grid[0] = 0
        self.grid_to_number[1] = 1
        self.number_to_grid[1] = 1
        for i in range(1, len(self.grid_unique)):
            self.grid_to_number[self.grid_unique[i]] = i + 1
            self.number_to_grid[i + 1] = self.grid_unique[i]

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
            self.userId.append(index)
            # 选取当前用户的轨迹
            current_df = df[df['userID'] == item]

            time_start = '2019-02-15 00:00:00'

            time_end = '2019-02-15 23:59:00'
            time_range = [[pd.to_datetime(time_start) + pd.Timedelta('{} Day'.format(index)),
                           pd.to_datetime(time_end) + pd.Timedelta('{} Day'.format(index))] for
                          index in range(4)]

            # 生成历史轨迹特征
            temp_hist_grid = []
            for item in time_range:
                current_grid = current_df[(current_df['time'] >= item[0]) & (current_df['time'] <= item[1])][
                    'number_grid'].values.reshape(-1).tolist()
                temp_hist_grid.append([item for item in current_grid])
                # temp_hist_grid.append([self.grid_to_number[item] for item in current_grid])
            self.hist_traj_grid.append(copy.deepcopy(temp_hist_grid))

            # 生成当前轨迹特征  随机用1去替换5个位置的坐标 保留去除的位置索引，
            current_time_start = pd.to_datetime('2019-02-19 00:00:00')
            current_time_end = pd.to_datetime('2019-02-19 23:59:00')
            current_grid = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'number_grid'].values.reshape(-1).tolist()
            curr_lats = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lat'].values.reshape(-1).tolist()
            curr_lngs = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lng'].values.reshape(-1).tolist()

            self.curr_traj_grid.append([item for item in copy.deepcopy(current_grid)])
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
            self.mask_value.append([item for item in temp_mask_value])
            self.mask_curr_traj_grid.append([item for item in current_grid])
            self.mask_value_lat.append(temp_mask_lat)
            self.mask_value_lng.append(temp_mask_lng)

            # 聚合历史轨迹
            temp_aggregator_hist_traj_grid = []
            temp_hist_grid = np.array(temp_hist_grid)
            for i in range(len(temp_hist_grid[0])):
                temp = temp_hist_grid[:, i].tolist()
                counters = [temp.count(item) for item in temp_hist_grid[:, i]]
                temp_aggregator_hist_traj_grid.append(temp[counters.index(max(counters))])

            self.aggregator_hist_traj_grid.append(temp_aggregator_hist_traj_grid)

        print(np.array(self.hist_traj_grid).shape, np.array(self.userId).shape, np.array(self.mask_pos).shape,
              np.array(self.mask_value).shape, np.array(self.curr_traj_grid).shape, np.array(self.dtw_border).shape,
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
        self.dtw_border = torch.Tensor(self.dtw_border)

    def __len__(self):
        return len(self.hist_traj_grid)

    def __getitem__(self, item):
        # 历史轨迹数据 用户号 位置掩码 当前掩码轨迹 y(真实值) 当前完整轨迹(没啥用 预留)
        return self.hist_traj_grid[item], self.userId[item], self.mask_pos[item], self.mask_curr_traj_grid[item], \
               self.mask_value[item], self.curr_traj_grid[item], self.dtw_border[item], self.mask_value_lat[item], \
               self.mask_value_lng[item]


class shenzhen_data_loader(Dataset):
    def __init__(self, path, mask, time_step):
        self.mask = mask
        self.path = path
        self.time_step = time_step
        self.__read_data__(mask)
        print('数据预处理结束')

    def __read_data__(self, mask_num):
        df = pd.read_csv(self.path, parse_dates=['time'], infer_datetime_format=True)
        dtw_df = pd.read_csv(
            'processed_data/shenzhen_taxi/shenzhen_texi_10_500_113.6139_114.5961_22.461_22.9998_dtw.csv')
        self.dtw_border = dtw_df['dtw'].values.reshape(-1).tolist()

        # df['time'] = df['time'].apply(lambda x: pd.to_datetime(x))
        # 获取所有用户
        users_id = np.unique(df['userID'].values.reshape(-1))

        # 获取网格数量
        self.grids = df['number_grid'].values.reshape(-1)
        print('有{}个用户'.format(len(users_id)))
        print('最大网格单元编码{},最小网格单元编号{}'.format(np.nanmax(self.grids), np.nanmin(self.grids)))

        self.user_count = len(users_id)
        # 缺失的位置用0代替
        self.grids[np.isnan(self.grids)] = 0
        self.grid_unique = np.unique(self.grids)
        self.grid_to_number = {}
        self.number_to_grid = {}
        self.grid_count = max(self.grid_unique) + 1
        self.grid_to_number[0] = 0
        self.number_to_grid[0] = 0
        self.grid_to_number[1] = 1
        self.number_to_grid[1] = 1
        for i in range(1, len(self.grid_unique)):
            self.grid_to_number[self.grid_unique[i]] = i + 1
            self.number_to_grid[i + 1] = self.grid_unique[i]

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
            self.userId.append(index)
            # 选取当前用户的轨迹
            current_df = df[df['userID'] == item]

            time_start = '2009-09-04 00:00:00'

            time_end = '2009-09-04 23:59:00'
            time_range = [[pd.to_datetime(time_start) + pd.Timedelta('{} Day'.format(index)),
                           pd.to_datetime(time_end) + pd.Timedelta('{} Day'.format(index))] for
                          index in range(4)]

            # 生成历史轨迹特征
            temp_hist_grid = []
            for item in time_range:
                current_grid = current_df[(current_df['time'] >= item[0]) & (current_df['time'] <= item[1])][
                    'number_grid'].values.reshape(-1).tolist()

                temp_hist_grid.append([item for item in current_grid[:self.time_step]])
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
            current_time_start = pd.to_datetime('2009-09-08 00:00:00')
            current_time_end = pd.to_datetime('2009-09-08 23:59:00')
            current_grid = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'number_grid'].values.reshape(-1).tolist()
            curr_lats = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lat'].values.reshape(-1).tolist()
            curr_lngs = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lng'].values.reshape(-1).tolist()

            self.curr_traj_grid.append([item for item in copy.deepcopy(current_grid)])
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
            self.mask_value.append([item for item in temp_mask_value])
            self.mask_curr_traj_grid.append([item for item in current_grid])
            self.mask_value_lat.append(temp_mask_lat)
            self.mask_value_lng.append(temp_mask_lng)

        print(np.array(self.hist_traj_grid).shape, np.array(self.userId).shape, np.array(self.mask_pos).shape,
              np.array(self.mask_value).shape, np.array(self.curr_traj_grid).shape, np.array(self.dtw_border).shape,
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
        self.dtw_border = torch.Tensor(self.dtw_border)

    def __len__(self):
        return len(self.hist_traj_grid)

    def __getitem__(self, item):
        # 历史轨迹数据 用户号 位置掩码 当前掩码轨迹 y(真实值) 当前完整轨迹(没啥用 预留)
        return self.hist_traj_grid[item], self.userId[item], self.mask_pos[item], self.mask_curr_traj_grid[item], \
               self.mask_value[item], self.curr_traj_grid[item], self.dtw_border[item], self.mask_value_lat[item], \
               self.mask_value_lng[item]

class chengdu_data_loader(Dataset):
    def __init__(self, path, mask, time_step):
        self.mask = mask
        self.path = path
        self.time_step = time_step
        self.__read_data__(mask)
        print('数据预处理结束')

    def __read_data__(self, mask_num):
        df = pd.read_csv(self.path, parse_dates=['time'], infer_datetime_format=True)
        dtw_df = pd.read_csv(
            'processed_data/chengdu_taxi/chengdu_10_500_103.2697_104.6097_30.2907_31.0325_dtw(1).csv')
        self.dtw_border = dtw_df['dtw'].values.reshape(-1).tolist()

        # df['time'] = df['time'].apply(lambda x: pd.to_datetime(x))
        # 获取所有用户
        users_id = np.unique(df['userID'].values.reshape(-1))

        # 获取网格数量
        self.grids = df['number_grid'].values.reshape(-1)
        print('有{}个用户'.format(len(users_id)))
        print('最大网格单元编码{},最小网格单元编号{}'.format(np.nanmax(self.grids), np.nanmin(self.grids)))

        self.user_count = len(users_id)
        # 缺失的位置用0代替
        self.grids[np.isnan(self.grids)] = 0
        self.grid_unique = np.unique(self.grids)
        self.grid_to_number = {}
        self.number_to_grid = {}
        self.grid_count = max(self.grid_unique) + 1
        self.grid_to_number[0] = 0
        self.number_to_grid[0] = 0
        self.grid_to_number[1] = 1
        self.number_to_grid[1] = 1
        for i in range(1, len(self.grid_unique)):
            self.grid_to_number[self.grid_unique[i]] = i + 1
            self.number_to_grid[i + 1] = self.grid_unique[i]

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
            self.userId.append(index)
            # 选取当前用户的轨迹
            current_df = df[df['userID'] == item]

            time_start = '2014-08-08 00:00:00'

            time_end = '2014-08-08 23:59:00'
            time_range = [[pd.to_datetime(time_start) + pd.Timedelta('{} Day'.format(index)),
                           pd.to_datetime(time_end) + pd.Timedelta('{} Day'.format(index))] for
                          index in range(4)]

            # 生成历史轨迹特征
            temp_hist_grid = []
            for item in time_range:
                current_grid = current_df[(current_df['time'] >= item[0]) & (current_df['time'] <= item[1])][
                    'number_grid'].values.reshape(-1).tolist()

                temp_hist_grid.append([item for item in current_grid[:self.time_step]])
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
            current_time_start = pd.to_datetime('2014-08-12 00:00:00')
            current_time_end = pd.to_datetime('2014-08-12 23:59:00')
            current_grid = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'number_grid'].values.reshape(-1).tolist()
            curr_lats = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lat'].values.reshape(-1).tolist()
            curr_lngs = \
                current_df[(current_df['time'] >= current_time_start) & (current_df['time'] <= current_time_end)][
                    'lng'].values.reshape(-1).tolist()

            self.curr_traj_grid.append([item for item in copy.deepcopy(current_grid)])
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
            self.mask_value.append([item for item in temp_mask_value])
            self.mask_curr_traj_grid.append([item for item in current_grid])
            self.mask_value_lat.append(temp_mask_lat)
            self.mask_value_lng.append(temp_mask_lng)

        print(np.array(self.hist_traj_grid).shape, np.array(self.userId).shape, np.array(self.mask_pos).shape,
              np.array(self.mask_value).shape, np.array(self.curr_traj_grid).shape, np.array(self.dtw_border).shape,
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
        self.dtw_border = torch.Tensor(self.dtw_border)

    def __len__(self):
        return len(self.hist_traj_grid)

    def __getitem__(self, item):
        # 历史轨迹数据 用户号 位置掩码 当前掩码轨迹 y(真实值) 当前完整轨迹(没啥用 预留)
        return self.hist_traj_grid[item], self.userId[item], self.mask_pos[item], self.mask_curr_traj_grid[item], \
               self.mask_value[item], self.curr_traj_grid[item], self.dtw_border[item], self.mask_value_lat[item], \
               self.mask_value_lng[item]

def construct_graph(inputs):
    # [512,48]
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    # 一天内最多去过max_n_node个区域
    max_n_node = np.max(n_node)

    # flag = 0
    for u_input in inputs:
        # unique自带排序
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])

        u_A = np.zeros((max_n_node, max_n_node))
        for i in range(len(u_input) - 1):
            if u_input[i + 1] == 0:
                continue
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1

        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)

        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)

        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)

        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

    # [250,48] [250,18,36] [250,18]
    # alias_inputs按照出现的位置从小到大进行位置编码
    return alias_inputs, A, items
