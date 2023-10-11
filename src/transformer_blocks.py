import math

import torch

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError('Embedding size need to be divisible by num_heads')

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.shape
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def attention_dot_product(self, query, keys, values, mask=None):
        attention_score = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, 1e-10)
        attention_probs = F.softmax(attention_score, dim=-1)
        outputs = torch.matmul(attention_probs, values)

        return outputs

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, query, key, value, mask=None):
        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))

        attention_product = self.attention_dot_product(query, key, value, mask)

        outputs = self.W_out(self.combine_heads(attention_product))

        return outputs


class FeedForward(nn.Module):
    def __init__(self, d_model: int, tmp_d: int):
        super().__init__()

        self.d_model = d_model
        self.tmp_d = tmp_d

        self.fc1 = nn.Linear(self.d_model, self.tmp_d)
        self.fc2 = nn.Linear(self.tmp_d, self.d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc2(self.activation(self.fc1(x)))
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length: int, d_model: int):
        super().__init__()

        positional_encoding = torch.zeros([max_seq_length, d_model])

        t = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        w_k = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

        positional_encoding[:, 0::2] = torch.sin(t * w_k)
        positional_encoding[:, 1::2] = torch.cos(t * w_k)

        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, tmp_d: int, dropout: float):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, tmp_d)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attention_output = self.attention(query=x, key=x, value=x, mask=src_mask)
        x = self.norm1(x + self.dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, tmp_d: int, dropout: float):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, tmp_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, target_mask):
        attention_output = self.attention(query=x, key=x, value=x, mask=target_mask)
        x = self.norm1(x + self.dropout(attention_output))

        attention_output = self.cross_attention(query=x, key=encoder_output, value=encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feed_forward_output))

        return x


class Encoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            tmp_d: int,
            src_vocab_size: int,
            max_seq_length: int,
            dropout: float
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(src_vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, tmp_d, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_mask):
        x = self.dropout(self.position_embedding(self.word_embedding(x)))
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            tmp_d: int,
            trg_vocab_size: int,
            max_seq_length: int,
            dropout
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, tmp_d, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.dropout(self.position_embedding(self.word_embedding(x)))
        for layer in self.layers:
            x = layer(x=x, encoder_output=encoder_output, src_mask=src_mask, target_mask=target_mask)
        x = self.fc_out(x)

        return x
