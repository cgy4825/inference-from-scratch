import torch
import torch.nn.functional as F

torch.manual_seed(0)

batch   = 1
seq_len = 4
d_model = 8
d_k     = d_model          # 先做单头

x = torch.randn(batch, seq_len, d_model)     # (1, 4, 8)
w_q = torch.randn(d_model, d_k)              # (8, 8)
w_k = torch.randn(d_model, d_k)              # (8, 8)
w_v = torch.randn(d_model, d_k)              # (8, 8)

Q = torch.matmul(x, w_q)                     # (1, 4, 8)
K = torch.matmul(x, w_k)                     # (1, 4, 8)
V = torch.matmul(x, w_v)                     # (1, 4, 8)

# 计算注意力分数
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (1, 4, 4)

mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(mask)
scores = scores.masked_fill(mask, float('-inf'))
# 计算注意力权重
attention_weights = F.softmax(scores, dim=-1)  # (1, 4, 4)

# 计算输出
output = torch.matmul(attention_weights, V)     # (1, 4, 8)

print("x      ", x.shape)          # (1, 4, 8)
print("Q      ", Q.shape)          # (1, 4, 8)
print("scores ", scores.shape)     # (1, 4, 4)
print("attn   ", attention_weights.shape)       # (1, 4, 4)
print("out    ", output.shape)        # (1, 4, 8)

# ① 每一行注意力权重加起来应该等于 1
print("每行和:", attention_weights.sum(dim=-1))   # 应该全是 1.0（或 1.0000）

# ② 打印一个注意力矩阵看看
print(attention_weights[0])