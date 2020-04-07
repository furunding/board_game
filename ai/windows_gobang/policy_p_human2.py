import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import os

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


# 1*1985*64*64
class Policy(nn.Module):
    def __init__(self, input_channel, grid_size):
        super(Policy, self).__init__()
        self.grid_size = grid_size

        self.conv = torch.nn.Sequential(
            nn.Conv2d(input_channel, 64, 5, stride=1, padding=2),  # n*64*15*15
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # n*128*15*15
            nn.ReLU(),
            nn.Conv2d(128, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),  # n*192*15*15
            nn.ReLU(),
            nn.Conv2d(192, 2, 1, stride=1, padding=0),  # n*2*15*15
            # nn.BatchNorm2d(2),
        )

        self.apply(weights_init)

    # inputs---obs,3*15*15, mask--[0,1] or [1, 0]  代表当前应该哪个选手落子
    def forward(self, inputs, mask):
        # 输入inputs batch*3*15*15
        x = self.conv(inputs)
        out = x.view(x.shape[0], x.shape[1], -1)   # batch * 2 * 225,(225代表棋盘大小)

        # out维度（batch, 2, 225）, 中间的维度2,分别表示2个选手落子的概率分布
        # out = torch.softmax(x, dim=2)

        # 将mask的维度转成跟out的维度一致
        mask = mask.unsqueeze(2)
        mask = mask.repeat(1, 1, self.grid_size*self.grid_size)
        # print(out)
        # print(mask)
        # 计算网络最终的输出
        out = out[:, 0, :]*mask[:, 0, :] + out[:, 1, :]*mask[:, 1, :]

        return out


def main():
    pass


#
if __name__ == '__main__':
    main()
