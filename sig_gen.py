import torch


class SignalGenerator:
    def __init__(self, config):
        self.config = config

        # 计算采样点数
        Na = int(round(config.Ra * config.Fa / config.Vr))  # 方位向采样点数
        Nr = int(round((2 * (config.Rmax - config.Rmin) / config.c + config.Tr) * config.Fr))  # 距离向采样点数

        # 创建慢时间轴 (方位向时间)
        eta_start = -config.Ra / (2 * config.Vr)
        eta_end = eta_start + (Na - 1) / config.Fa
        self.eta = torch.linspace(eta_start, eta_end, Na, dtype=torch.float64)

        # 创建快时间轴 (距离向时间)
        tao_start = 2 * config.Rmin / config.c - config.Tr / 2
        tao_end = tao_start + (Nr - 1) / config.Fr
        self.tau = torch.linspace(tao_start, tao_end, Nr, dtype=torch.float64)

        # 平台位置 (沿航迹方向)
        self.y = config.Vr * self.eta

        # 更新配置参数
        config.Na = Na  # 方位向采样点数
        config.Nr = Nr  # 距离向采样点数

    def rectpuls(self, t, width):
        return torch.where((t >= -width / 2) & (t <= width / 2),
                           torch.ones_like(t, dtype=torch.complex128),
                           torch.zeros_like(t, dtype=torch.complex128))

    def generate_signal(self, targets):
        config = self.config
        # 初始化接收信号矩阵
        signal_receive = torch.zeros((config.Na, config.Nr), dtype=torch.complex128)
        targets_tensor = torch.tensor(targets, dtype=torch.float64)

        # 存储每个目标在每个慢时间时刻的斜距
        r_eta = torch.zeros((len(targets), config.Na), dtype=torch.float64)

        # 生成每个目标的回波信号
        for i in range(len(targets)):
            target_x, target_y = targets_tensor[i, 0], targets_tensor[i, 1]
            # 计算目标在慢时间轴上的瞬时斜距
            r_eta[i, :] = torch.sqrt(target_x ** 2 + (target_y - self.y) ** 2 + config.H ** 2)

            # 遍历每个慢时间点
            for j in range(config.Na):
                # 方位向照射范围约束 (仅处理在波束照射范围内的目标)
                if torch.abs(target_y - self.y[j]) >= config.Ls / 2:
                    continue

                # 计算双程延迟时间
                delay = 2 * r_eta[i, j] / config.c

                # 生成距离向矩形包络
                rect_pulse = self.rectpuls(self.tau - delay, config.Tr)

                # 计算相位项 (载频相位 + 线性调频相位)
                amplitude = torch.exp(-1j * 4 * torch.pi * config.f0 * r_eta[i, j] / config.c)
                chirp_phase = torch.exp(1j * torch.pi * config.Kr * (self.tau - delay) ** 2)

                # 累加到接收信号矩阵
                signal_receive[j, :] += rect_pulse * amplitude * chirp_phase

        return signal_receive
