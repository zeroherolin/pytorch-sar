import torch
import torch.nn as nn
import time

from params import Config
from sig_gen import SignalGenerator
from visualize import visualize


class WKA(nn.Module):
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        print(f"Initializing optimized WKA processor on {device}...")

        # 将参数转移到device
        self.Vr = torch.tensor(config.Vr, dtype=torch.float64, device=self.device)
        self.theta_c = torch.tensor(config.theta_c, dtype=torch.float64, device=self.device)
        self.lamda = torch.tensor(config.lamda, dtype=torch.float64, device=self.device)
        self.Fr = torch.tensor(config.Fr, dtype=torch.float64, device=self.device)
        self.Fa = torch.tensor(config.Fa, dtype=torch.float64, device=self.device)
        self.R_ref = torch.tensor(config.R_ref, dtype=torch.float64, device=self.device)
        self.c = torch.tensor(config.c, dtype=torch.float64, device=self.device)
        self.f0 = torch.tensor(config.f0, dtype=torch.float64, device=self.device)
        self.Kr = torch.tensor(config.Kr, dtype=torch.float64, device=self.device)

        # 创建距离频率轴 (f_tau) 和方位频率轴 (f_eta)
        f_eta_ref = 2 * self.Vr * torch.sin(self.theta_c) / self.lamda
        f_tau = torch.linspace(-config.Fr / 2, config.Fr / 2, config.Nr, dtype=torch.float64, device=device)
        f_eta = torch.linspace(-config.Fa / 2, config.Fa / 2, config.Na, dtype=torch.float64, device=device) + f_eta_ref

        # 扩展为矩阵用于向量化计算
        self.f_tau = f_tau.repeat(config.Na, 1) # [Na, Nr]
        self.f_eta = f_eta.reshape(-1, 1).repeat(1, config.Nr) # [Na, Nr]

        # 预计算二维频域匹配滤波器
        self.h2df = torch.exp(1j * (4 * torch.pi * self.R_ref / self.c * torch.sqrt(
            (self.f0 + self.f_tau) ** 2 - self.c ** 2 * self.f_eta ** 2 / (
                        4 * self.Vr ** 2)) + torch.pi * self.f_tau ** 2 / self.Kr))

    @staticmethod
    def fft2d_torch(x):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    @staticmethod
    def ifft2d_torch(x):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    def matched_filtering_torch(self, sig):
        if sig.device != self.h2df.device:
            sig = sig.to(self.h2df.device)
        s2df = sig * self.h2df
        return s2df

    def stolt_interpolation_torch(self, sig):
        Pi = 6  # 插值窗口宽度

        # 计算Stolt映射关系
        f_tao_pie = torch.sqrt((self.f0 + self.f_tau) ** 2 -
                               (self.c * self.f_eta / (2 * self.Vr)) ** 2) - self.f0

        # 计算采样点偏移量
        delta_n = (f_tao_pie - self.f_tau) * (self.config.Nr / self.Fr)

        # 创建插值偏移索引
        offsets = torch.arange(-Pi // 2, Pi // 2, dtype=torch.float64, device=self.device)

        # 扩展维度用于广播计算
        delta_n_exp = delta_n.unsqueeze(2) # [Na, Nr, 1]
        offsets_exp = offsets.view(1, 1, -1) # [1, 1, Pi]

        # 计算所有位置的索引
        indices = torch.arange(self.config.Nr, device=self.device).view(1, -1, 1) + offsets_exp # [1, Nr, Pi]

        # 边界处理：越界索引设为边界值
        indices = indices.clamp(0, self.config.Nr - 1)

        # 计算sinc权重
        diff = delta_n_exp - offsets_exp # [Na, Nr, Pi]
        sinc_weights = torch.sinc(diff)

        # 执行向量化插值，分批次处理方位向以避免内存溢出
        stolt_2df = torch.zeros((self.config.Na, self.config.Nr), dtype=torch.complex128, device=self.device)
        batch_size = 32  # 根据GPU内存调整
        for m_start in range(0, self.config.Na, batch_size):
            m_end = min(m_start + batch_size, self.config.Na)
            m_slice = slice(m_start, m_end)

            # 获取当前批次的信号切片 [batch, Nr]
            sig_batch = sig[m_slice, :]

            # 为当前批次收集插值样本 [batch, Nr, Pi]
            samples = torch.gather(
                sig_batch.unsqueeze(2).expand(-1, -1, Pi + 1),
                1,
                indices.expand(m_end - m_start, -1, -1).long()
            )

            # 应用权重 [batch, Nr, Pi]
            weighted_samples = samples * sinc_weights[m_slice, :, :]

            # 沿偏移维度求和 [batch, Nr]
            stolt_batch = weighted_samples.sum(dim=2)

            stolt_2df[m_slice, :] = stolt_batch

        # 参考距离平移
        phase_comp = torch.exp(-1j * 4 * torch.pi * self.R_ref / self.c * self.f_tau)
        stolt_2df *= phase_comp

        return stolt_2df

    def forward(self, sig):
        print("\nStarting optimized WKA processing...")
        start_time = time.time()

        if sig.device != self.device:
            sig = sig.to(self.device)

        # 1. 二维FFT变换到频域
        print("Performing FFT 2D...")
        s2df = self.fft2d_torch(sig)
        print(f"2D FFT completed in {time.time() - start_time:.2f}s")

        # 2. 频域匹配滤波
        print("Performing Matched Filtering...")
        s2df_matched = self.matched_filtering_torch(s2df)
        print(f"Matched filtering completed in {time.time() - start_time:.2f}s")

        # 3. Stolt插值
        print("Performing Stolt Iinterpolation...")
        stolt_start = time.time()
        s2df_stolt = self.stolt_interpolation_torch(s2df_matched)
        print(f"Stolt interpolation completed in {time.time() - stolt_start:.2f}s")

        # 4. 二维IFFT到图像
        print("Performing IFFT 2D...")
        image = self.ifft2d_torch(s2df_stolt)
        print(f"Total processing time: {time.time() - start_time:.2f}s")

        return s2df, s2df_matched, s2df_stolt, image


if __name__ == "__main__":
    config = Config()
    generator = SignalGenerator(config)

    targets = [
        (config.Xc, config.Yc), # 场景中心
        (config.Xc - 300, config.Yc + 100),
        (config.Xc - 300, config.Yc - 200),
        (config.Xc - 100, config.Yc - 100)
    ]

    print("Target positions:")
    for i, (x, y) in enumerate(targets):
        print(f"  Target {i + 1}: x={x:.2f}m, y={y:.2f}m")

    print("\nGenerating SAR signal...")
    start_time = time.time()
    sar_signal = generator.generate_signal(targets)
    gen_time = time.time() - start_time
    print(f"Signal generation completed in {gen_time:.2f} seconds")

    # 处理信号
    wka = WKA(config)
    s2df, s2df_matched, s2df_stolt, image = wka.forward(sar_signal)

    # 转换结果为NumPy数组
    sar_signal_np = torch.abs(sar_signal).detach().cpu().numpy()
    s2df_matched_np = torch.abs(WKA.ifft2d_torch(s2df_matched)).detach().cpu().numpy()
    image_np = torch.abs(image).detach().cpu().numpy()

    # 可视化函数绘图
    visualize([sar_signal_np, s2df_matched_np, image_np],
              ['Original Echo Signal', 'Matched Filtered Signal', 'Stolt Interpolated Signal'])
