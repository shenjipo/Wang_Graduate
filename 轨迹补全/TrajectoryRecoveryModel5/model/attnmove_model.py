import torch
import torch.nn as nn

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


class AttnMoveModel(nn.Module):
    def __init__(self, d_model, heads, grid_count, user_count, dropout):
        super(AttnMoveModel, self).__init__()
        self.d_model = d_model
        self.grid_embedding = nn.Embedding(int(grid_count), d_model)
        self.pe = PositionalEmbedding(d_model=d_model)
        self.cross_attention = SelfAttention(d_model, heads=heads)
        self.liner_hist = nn.Linear(in_features=d_model, out_features=d_model)
        self.self_attention_curr = SelfAttention(d_model, heads=heads)
        self.self_attention_hist = SelfAttention(d_model, heads=heads)
        self.liner_transform = nn.Linear(d_model * 2, d_model)
        self.liner_transform2 = nn.Linear(d_model, d_model)

    def forward(self, hist_traj_grid, userId, mask_pos, mask_curr_traj_grid, aggregator_hist_traj_grid):
        # embedding
        # [batch_size,hist_size,seq_len] -> [batch_size,hist_size,seq_len,d_model]
        # [batch_size,seq_len] -> [batch_size,seq_len,d_model]
        aggregator_hist_traj_grid = self.grid_embedding(aggregator_hist_traj_grid)
        mask_curr_traj_grid = self.grid_embedding(mask_curr_traj_grid)
        aggregator_hist_traj_grid = aggregator_hist_traj_grid + self.pe(aggregator_hist_traj_grid)
        mask_curr_traj_grid = mask_curr_traj_grid + self.pe(mask_curr_traj_grid)

        '''
        hist self attention
        '''
        aggregator_hist_traj_grid = self.self_attention_hist(aggregator_hist_traj_grid, aggregator_hist_traj_grid,
                                                             aggregator_hist_traj_grid)

        '''
        curr self attention
        '''
        mask_curr_traj_grid = self.self_attention_curr(mask_curr_traj_grid, mask_curr_traj_grid, mask_curr_traj_grid)

        '''
        cross attention
        '''
        hist_traj_grid = self.cross_attention(mask_curr_traj_grid, aggregator_hist_traj_grid, aggregator_hist_traj_grid)

        # hist_traj_grid = self.dilation_cnn(
        #     hist_traj_grid.reshape(batch_size, -1, self.d_model).transpose(1, 2)).transpose(1, 2)

        '''
        trajectory recovery
        '''

        # [128,144,128]
        hybrid_embedding = self.liner_transform(torch.cat([hist_traj_grid, mask_curr_traj_grid], -1))
        hybrid_embedding = self.liner_transform2(torch.tanh(mask_curr_traj_grid))

        # hybrid_embedding = self.dropout(hybrid_embedding)

        index = mask_pos + torch.arange(mask_pos.shape[0]).to(mask_pos.device).view(-1, 1) * hybrid_embedding.shape[1]
        index = index.view(-1)
        # [1280,128]
        hybrid_embedding = hybrid_embedding.view(-1, self.d_model).index_select(0, index)

        candidate_poi = self.grid_embedding.weight[2:]
        # scores: batch_size * mask_num, n
        scores = torch.matmul(hybrid_embedding, candidate_poi.transpose(1, 0))

        score = F.log_softmax(scores, dim=-1)

        # [1280,21352]
        return score


class MultiHeadCrossAttention_my(nn.Module):
    def __init__(self, d_model, heads):
        super(MultiHeadCrossAttention_my, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.liner_q = nn.Linear(d_model, d_model)
        self.liner_k = nn.Linear(d_model, d_model)
        self.liner_v = nn.Linear(d_model, d_model)
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
