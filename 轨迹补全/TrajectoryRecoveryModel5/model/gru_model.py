import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(-2)]


class MyGRU(nn.Module):
    def __init__(self, d_model, heads, grid_count, user_count, time_step):
        super(MyGRU, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.grid_count = grid_count
        self.user_count = user_count
        self.time_step = time_step
        self.grid_embedding = nn.Embedding(int(grid_count), d_model)
        self.pe = PositionalEmbedding(d_model=d_model)
        self.gru_hist = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, bias=True, batch_first=True)
        self.gru_curr = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, bias=True, batch_first=True)
        self.liner_transform = nn.Linear(d_model * 2, d_model)
        self.liner_transform2 = nn.Linear(d_model, d_model)

    def forward(self, hist_traj_grid, userId, mask_pos, mask_curr_traj_grid, aggregator_hist_traj_grid):
        # 历史轨迹序列 用户编号序列 掩码位置序列 掩码当前轨迹序列 历史轨迹图 当前轨迹图
        # 历史经过位置序列(不重复) 当前经过的位置序列(不重复)    一共8个参数

        batch_size, hist_size, seq_size, = hist_traj_grid.shape
        hist_traj_grid = hist_traj_grid.reshape(batch_size, -1)

        hist_traj_grid = self.grid_embedding(hist_traj_grid)
        mask_curr_traj_grid = self.grid_embedding(mask_curr_traj_grid)

        hist_traj_grid = hist_traj_grid + self.pe(hist_traj_grid)
        mask_curr_traj_grid = mask_curr_traj_grid + self.pe(mask_curr_traj_grid)

        out_hist, h0_hist = self.gru_hist(hist_traj_grid)
        hist_traj_grid = out_hist[:, -self.time_step:, :]
        out_curr, h0_curr = self.gru_hist(mask_curr_traj_grid)
        mask_curr_traj_grid = out_curr


        hybrid_embedding = self.liner_transform(torch.cat([hist_traj_grid, mask_curr_traj_grid], -1))
        hybrid_embedding = self.liner_transform2(torch.tanh(hybrid_embedding))

        index = mask_pos + torch.arange(mask_pos.shape[0]).to(mask_pos.device).view(-1, 1) * hybrid_embedding.shape[1]
        index = index.view(-1)
        # [1280,128]
        hybrid_embedding = hybrid_embedding.view(-1, self.d_model).index_select(0, index)

        candidate_poi = self.grid_embedding.weight[2:]

        scores = torch.matmul(hybrid_embedding, candidate_poi.transpose(1, 0))

        score = F.log_softmax(scores, dim=-1)

        return score

