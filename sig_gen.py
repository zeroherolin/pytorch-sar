import math
import torch
from numpy import kaiser

from params import Config

# 创建Kaiser窗 (w1)
w1 = torch.tensor(kaiser(Config().Nrg, beta=2.5), dtype=torch.float64)
w1 = w1.repeat(Config().Naz, 1)


class SignalGenerator:
    def __init__(self, config):
        self.config = config
        print("Initializing signal generator...")

        # 计算时间轴参数
        dr = 1 / config.Fr
        da = 1 / config.Fa

        # 创建距离时间轴 (tr)
        tr = torch.arange(-config.Nrg / 2, config.Nrg / 2, dtype=torch.float64) * dr + 2 * config.Rref / config.c
        tr = tr.repeat(config.Naz, 1)

        # 创建方位时间轴 (ta)
        ta = torch.arange(0, config.Naz, dtype=torch.float64) * da
        ta = ta.reshape(-1, 1).repeat(1, config.Nrg)

        self.tr = tr
        self.ta = ta

    def generate_signal(self, targets):
        config = self.config

        # 初始化信号矩阵
        sig = torch.zeros((config.Naz, config.Nrg), dtype=torch.complex128)

        # 生成每个目标的信号
        for x, y in targets:
            # 计算斜距历史
            r = torch.sqrt(torch.tensor(x ** 2, dtype=torch.float64) +
                           (torch.tensor(y, dtype=torch.float64) - config.Vr * self.ta) ** 2)

            # 计算双程延迟
            delay = 2 * r / config.c

            # 计算脉冲存在的时间范围
            limit = torch.abs(self.tr - delay) < config.Tr / 2

            # 计算线性调频信号和相位项
            chirp = torch.exp(1j * math.pi * config.Kr * (self.tr - delay) ** 2)
            phase = torch.exp(-1j * 4 * math.pi * config.fc * r / config.c)

            # 叠加目标信号
            sig += w1 * limit * chirp * phase

        return sig
