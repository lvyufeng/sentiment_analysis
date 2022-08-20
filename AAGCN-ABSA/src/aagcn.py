# -*- coding: utf-8 -*-
import numpy
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common.initializer import Normal


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32,
                 padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size,
                         use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings,
                        padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze

        return embedding


class GraphConvolution(nn.Cell):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mindspore.Parameter(
            ops.zeros((in_features, out_features), mindspore.float32))
        if bias:
            self.bias = mindspore.Parameter(
                ops.zeros((out_features), mindspore.float32))
        else:
            self.bias = None

    def construct(self, text, adj):
        # torch.matmul对应ops.matmul
        output = ops.matmul(text, self.weight)
        denom = adj.sum(axis=2, keepdims=True) + 1
        output = ops.matmul(adj, output)
        # 使用ms的Tensor.sum() 传参为axis和keepdims 和torch不用
        output = output / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class AAGCN(nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(AAGCN, self).__init__()
        self.opt = opt
        self.embed = Embedding.from_pretrained_embedding(
            Tensor(embedding_matrix, dtype=mindspore.float32))
        self.text_lstm = nn.LSTM(
            opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        # mindspore.nn.Dense等价torch.nn.Linear
        self.fc = nn.Dense(2*opt.hidden_dim, opt.polarities_dim)
        # Dropout也可以等价替换
        self.text_embed_dropout = nn.Dropout(1 - 0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(2)

    def construct(self, text_indices, adj, d_adj, seq_length):
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, _ = self.text_lstm(text, seq_length=seq_length)
        x = self.gc1(text_out, adj)
        x = self.relu(x)

        x = self.relu(self.gc2(x, d_adj))
        x = self.relu(self.gc3(x, adj))
        x = self.relu(self.gc4(x, d_adj))

        alpha_mat = ops.matmul(x, text_out.transpose(0, 2, 1))
        alpha = self.softmax(alpha_mat.sum(axis=1, keepdims=True))
        x = ops.matmul(alpha, text_out).squeeze(1)

        output = self.fc(x)
        return output