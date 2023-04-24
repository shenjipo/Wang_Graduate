import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


class GGNN(nn.Module):
    def __init__(self, d_model, step):
        super(GGNN, self).__init__()
        self.d_model = d_model
        self.step = step
        self.linear_in = nn.Linear(self.d_model, self.d_model)
        self.linear_out = nn.Linear(self.d_model, self.d_model)
        self.linear_asz = nn.Linear(self.d_model * 2, self.d_model)
        self.linear_asr = nn.Linear(self.d_model * 2, self.d_model)
        self.linear_uz = nn.Linear(self.d_model, self.d_model)
        self.linear_ur = nn.Linear(self.d_model, self.d_model)

        self.linear_wh = nn.Linear(self.d_model * 2, self.d_model)
        self.linear_uo = nn.Linear(self.d_model, self.d_model)

    def gnn_cell(self, A, hidden):
        batch_size, location_step, _ = A.shape
        # [250,24,24]
        Mi = A[:, :, :location_step]
        # [250,24,24]
        Mo = A[:, :, location_step:]
        # [250,24,24]*[250,24,128] = [250,24,128]
        asi = torch.matmul(Mi, self.linear_in(hidden))
        aso = torch.matmul(Mo, self.linear_out(hidden))
        # [250,24,256]
        As = torch.cat([asi, aso], dim=-1)
        # [250,24,128] + [250,24,128] = [250,24,128]
        Zs = torch.sigmoid(self.linear_asz(As) + self.linear_uz(hidden))
        Rs = torch.sigmoid(self.linear_asr(As) + self.linear_ur(hidden))

        e_ = torch.tanh(self.linear_wh(As) + self.linear_uo(torch.mul(Rs, hidden)))
        e = torch.mul(1 - Zs, hidden) + torch.mul(Zs, e_)

        return e

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.gnn_cell(A, hidden)
        return hidden


class SoftAttention(nn.Module):
    def __init__(self, d_model):
        super(SoftAttention, self).__init__()
        self.liner_q = nn.Linear(d_model, d_model)
        self.liner_k = nn.Linear(d_model, d_model)
        self.liner_three = nn.Linear(d_model, 1)

    def forward(self, q, k):
        # query是历史信息 key是当前信息
        query = self.liner_q(q)
        key = self.liner_k(k).unsqueeze(1)

        # [32,48,128]
        # [32,4,48,128]

        attn_weights = self.liner_three(torch.sigmoid(query + key))
        attn_value = attn_weights * q
        attn_value = attn_value.sum(1)

        return attn_value


class MultiHeadCrossAttention_my(nn.Module):
    def __init__(self, d_model, heads):
        super(MultiHeadCrossAttention_my, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.liner_q = nn.Linear(d_model, d_model)
        self.liner_k = nn.Linear(d_model, d_model)
        self.liner_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.compress_layer = nn.Linear(d_model * heads, d_model)

    def forward(self, q, k, v):
        batch_size, hist_size, seq_len, d_model = k.shape

        # q batch_size,seq_len,d_model -> batch_size,seq_len,d_model*heads
        query = self.liner_q(q)
        key = self.liner_k(k)
        value = self.liner_v(v)
        # batch_size, seq_len, d_model * heads -> batch_size*heads, seq_len, d_model
        # batch_size, hist_size, seq_len, d_model * heads -> batch_size*heads, hist_size, seq_len, d_model
        query = torch.cat(query.split(self.d_model // self.heads, dim=-1), dim=0)
        key = torch.cat(key.split(self.d_model // self.heads, dim=-1), dim=0)
        value = torch.cat(value.split(self.d_model // self.heads, dim=-1), dim=0)

        key = key.reshape(batch_size * self.heads, -1, self.d_model // self.heads).transpose(1, 2)
        # batch_size*heads,seq_len,seq_len*hist_size
        attn_weights = torch.matmul(query, key).reshape(batch_size * self.heads, seq_len, hist_size, seq_len)
        attn_weights = torch.softmax(attn_weights, -1)

        attn_values = torch.matmul(attn_weights.transpose(1, 2), value)

        attn_values = torch.cat(attn_values.split(batch_size, 0), -1)

        return attn_values


class SelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.liner_q = nn.Linear(d_model, d_model)
        self.liner_k = nn.Linear(d_model, d_model)
        self.liner_v = nn.Linear(d_model, d_model)

        self.liner_res = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.shape
        query = self.liner_q(q)
        key = self.liner_k(k)
        value = self.liner_v(v)
        # [32,48,128] 暂时不要多头
        query = torch.cat(query.split(self.d_model // self.heads, dim=-1), dim=0)
        key = torch.cat(key.split(self.d_model // self.heads, dim=-1), dim=0)
        value = torch.cat(value.split(self.d_model // self.heads, dim=-1), dim=0)

        key = key.reshape(batch_size * self.heads, -1, self.d_model // self.heads).transpose(1, 2)
        attn_weights = torch.matmul(query, key).reshape(batch_size * self.heads, seq_len, seq_len)

        attn_weights = torch.softmax(attn_weights, -1)
        attn_value = torch.matmul(attn_weights, value)
        attn_value = torch.cat(attn_value.split(batch_size, 0), -1)

        return attn_value


class MyModel(nn.Module):
    def __init__(self, d_model, heads, grid_count, user_count, time_step, step):
        super(MyModel, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.grid_count = grid_count
        self.user_count = user_count
        self.time_step = time_step
        self.grid_embedding = nn.Embedding(int(grid_count), d_model)
        self.pe = PositionalEmbedding(d_model=d_model)
        self.GNN = GGNN(d_model, step=step)
        self.cross_attention = MultiHeadCrossAttention_my(d_model, heads)
        self.soft_attention = SoftAttention(d_model)
        self.liner_transform = nn.Linear(d_model * 2, d_model)
        self.liner_transform2 = nn.Linear(d_model, d_model)
        # self.curr_self_attention = SelfAttention(d_model, heads=heads)
        self.hist_self_attention = SelfAttention(d_model, heads=heads)

        # self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.d_model)
        for weight in self.parameters():
            weight.data.uniform_(-1, 1)

    def forward(self, hist_traj_grid, dtw_border, mask_pos, mask_curr_traj_grid, historical_input_A, current_input_A,
                historical_inputs, curr_inputs, hist_traj_in_grid, curr_traj_in_grid):
        # 历史轨迹序列 用户编号序列 掩码位置序列 掩码当前轨迹序列 历史轨迹图 当前轨迹图
        # 历史经过位置序列(不重复) 当前经过的位置序列(不重复)    一共8个参数

        batch_size, hist_size, seq_size, = hist_traj_grid.shape

        # historical_inputs [512,46]
        # 128,46

        # historical_inputs = [512,46]
        historical_sessions = self.grid_embedding(historical_inputs)
        # historical_sessions = [512,46,128]     historical_input_A = [512,46,92]

        historical_sessions = self.GNN(historical_input_A, historical_sessions)

        # historical_sessions = [512,47,128]
        get_history = lambda i: historical_sessions[i][hist_traj_in_grid[i]]
        # seq_hidden_history [512,48,128]
        seq_hidden_history = torch.stack([get_history(i) for i in torch.arange(len(hist_traj_in_grid)).long()])

        seq_hidden_history = seq_hidden_history + self.pe(seq_hidden_history)
        # batch_size * history_size, seq_length, hidden_size -> batch_size, history_size, seq_length, hidden_size
        seq_hidden_history = seq_hidden_history.view(batch_size, -1, self.time_step, self.d_model)

        current_sessions = self.grid_embedding(curr_inputs)
        current_sessions = self.GNN(current_input_A, current_sessions)
        get_current = lambda i: current_sessions[i][curr_traj_in_grid[i]]
        # batch_size, seq_length, hidden_size
        seq_hidden_current = torch.stack([get_current(i) for i in torch.arange(len(curr_traj_in_grid)).long()])

        
        seq_hidden_current = seq_hidden_current + self.pe(seq_hidden_current)
        # curr self attention
        seq_hidden_history = self.cross_attention(seq_hidden_current, seq_hidden_history, seq_hidden_history)
        # seq_hidden_current = self.curr_self_attention(seq_hidden_current, seq_hidden_current, seq_hidden_current)

        '''
        hist dtw self attention
        '''
        
        seq_hidden_history = seq_hidden_history.reshape(batch_size, -1, self.d_model)
        seq_hidden_history = torch.stack(
          [seq_hidden_history[item, seq_size * dtw_border[item]:seq_size * (dtw_border[item] + 1), :] for item in
           range(len(dtw_border))], dim=0)
        '''
        trajectory recovery
        '''
        # [128,144,128]
        # hybrid_embedding = self.liner_transform(torch.cat([seq_hidden_history, seq_hidden_current], -1))
        hybrid_embedding = self.liner_transform2(torch.tanh(seq_hidden_history))
      
        index = mask_pos + torch.arange(mask_pos.shape[0]).to(mask_pos.device).view(-1, 1) * hybrid_embedding.shape[1]
        index = index.view(-1)
        # [1280,128]
        hybrid_embedding = hybrid_embedding.view(-1, self.d_model).index_select(0, index)

        candidate_poi = self.grid_embedding.weight[2:]
        # scores: batch_size * mask_num, n
        scores = torch.matmul(hybrid_embedding, candidate_poi.transpose(1, 0))

        score = F.log_softmax(scores, dim=-1)

        return score
