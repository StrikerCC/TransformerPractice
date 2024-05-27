import torch
from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        return


class Encoder(nn.Module):
    def __init__(self, c_in=64, c_mid=128, c_out=64):
        super().__init__()
        self.Wq, self.Wk, self.Wv = torch.Tensor(c_in, c_mid), torch.Tensor(c_in, c_mid), torch.Tensor(c_in, c_mid)
        self.norm0 = nn.LayerNorm(c_in)
        self.linear = torch.nn.Linear(c_in, c_out)
        self.act = torch.nn.GELU()
        self.norm1 = nn.LayerNorm(c_out)
        return

    def forward(self, x):
        q, k, v = torch.matmul(x, self.Wq), torch.matmul(x, self.Wk), x # torch.matmul(x, self.Wv)
        # print('q', q.shape)

        mat_a = torch.matmul(q, k.permute(0, 2, 1).contiguous()) / torch.norm(v) ** 0.5
        mat_a = torch.softmax(mat_a, dim=0)
        # print('mat_a', mat_a.shape)

        v_att = torch.matmul(mat_a, v)
        v_att = self.norm0(v_att)
        # print('v_att', v_att.shape)

        v_dim = self.linear(v_att)
        # print('v_dim', v_dim.shape)

        y = self.act(v_dim)
        y = self.norm1(y)
        return y


class Decoder(nn.Module):
    def __init__(self, c_in=64, c_mid=128, c_out=64):
        super().__init__()
        self.Wq, self.Wk, self.Wv = torch.Tensor(c_in, c_mid), torch.Tensor(c_in, c_mid), torch.Tensor(c_in, c_mid)
        self.linear = torch.nn.Linear(c_in, c_out)
        self.act = torch.nn.GELU()
        return

    def forward(self, x):
        return
