# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

"""
输入3个句子，每个句子由5个单词构成，每个单词词向量10维
batch=3, seq_len=5, Embedding=10
"""
# 设置LSTM参数，词向量维数10，隐藏元维度20,2个LSTM隐藏层，双向LSTM
bilstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)

# 如下表示输入句子
input = torch.randn(5, 3, 10)
# 初始化的隐藏元和记忆元，通常维度一样
h0 = torch.randn(4, 3, 20)  # [bidirection*num_layers, batch_size, hidden_size]
c0 = torch.randn(4, 3, 20)  # [bidirection*num_layers, batch_size, hidden_size]

# 这里有2层lstm，output是最后一层lstm的每个词向量对应隐藏层的输出，与层数无关，只与序列长度有关
output, (hn, cn) = bilstm(input, (h0, c0))
print("output shape：", output.shape)  # shape：torch.Size([5,3,40]),[seq_len,batch_size,2*hidden_size]
print("hn shape：", hn.shape)  # shape：torch.Size([4,3,20]),[bidirection*num_layers,batch_size,hidden_size]
print("cn shape：", cn.shape)  # shape：torch.Size([4,3,20]),[bidirection*num_layers,batch_size,hidden_size]

# 将输出数据做一个二分类
output = output.permute(1, 0, 2)  # torch.Size([3,5,40]),[batch_size,seq_len,2*hidden_size]
output = output.contiguous()	# torch.view()前做了permute需要contiguous，因为view需要tensor在连续的内存
batch_size = output.size(0)
output = output.view(batch_size, -1)  # torch.Size([3,200]),[batch_size,seq_len*2*hidden_size]
fully_connected = nn.Linear(200, 2)
output = fully_connected(output)
print(output.shape)  # torch.Size([3,2]),[batch_size,class]
print(output)
