import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import sys

from params import Config
from sig_gen import SignalGenerator, w1


class WKA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print("Initializing WKA processor...")

        # 创建距离频率轴 (fr) 和方位频率轴 (fa)
        fr = torch.arange(-config.Nrg / 2, config.Nrg / 2, dtype=torch.float64) * config.Fr / config.Nrg
        fa = torch.arange(-config.Naz / 2, config.Naz / 2, dtype=torch.float64) * config.Fa / config.Naz + config.fac
        self.fr = fr.repeat(config.Naz, 1)
        self.fa = fa.reshape(-1, 1).repeat(1, config.Nrg)

        # 二维频域匹配滤波器
        self.h2df = torch.exp(
            1j * (4 * torch.pi * config.Rref / config.c *
                  torch.sqrt((config.fc + self.fr) ** 2 - config.c ** 2 * self.fa ** 2 / (4 * config.Vr ** 2)) +
                  torch.pi * self.fr ** 2 / config.Kr)
        )

        print(f"Filter shape: {self.h2df.shape}")
        print(f"Filter min amplitude: {torch.abs(self.h2df).min().item()}")
        print(f"Filter max amplitude: {torch.abs(self.h2df).max().item()}")

    @staticmethod
    def fft2d_torch(x):
        print("Performing 2D FFT...")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    @staticmethod
    def ifft2d_torch(x):
        print("Performing 2D IFFT...")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    def matched_filtering_torch(self, sig):
        print("Applying matched filter...")
        s2df = sig * self.h2df
        return s2df

    def stolt_interpolation_torch(self, sig):
        print("Performing Stolt interpolation...")
        config = self.config
        Pi = 8 # 插值窗口宽度

        # 计算 Stolt 插值的变换坐标
        D_fr = torch.sqrt((self.fr + config.fc) ** 2 + (self.fa * config.c / (2 * config.Vr)) ** 2) - (self.fr + config.fc)
        Ndkx = torch.round(D_fr * config.Nrg / config.Fr).to(torch.int)
        Deci = D_fr * config.Nrg / config.Fr - Ndkx
        Nd = torch.max(torch.abs(Ndkx)).item()

        # 创建填充信号
        pad_left_part = sig[:, - (Nd + Pi + 1):]
        pad_right_part = sig[:, :Nd + Pi]
        pad_sig = torch.cat([pad_left_part, sig, pad_right_part], dim=1)

        # 初始化 Stolt 插值结果
        stolt = torch.zeros((config.Naz, config.Nrg), dtype=torch.complex128)

        # 执行 Stolt 插值
        for m in range(config.Naz):
            for n in range(Pi - 1, config.Nrg):
                # 生成 sinc 插值核
                h = torch.sinc(torch.arange(Deci[m, n].item() + Pi / 2 - 1, Deci[m, n].item() - Pi / 2 - 1, -1, dtype=torch.float64))

                # 提取窗口数据
                start_idx = n + Nd + Pi + Ndkx[m, n].item() - Pi // 2
                end_idx = n + Nd + Pi + Ndkx[m, n].item() + Pi // 2
                d = pad_sig[m, start_idx:end_idx]

                # 应用插值
                stolt[m, n] = torch.dot(torch.complex(h, torch.zeros(h.shape, dtype=torch.float64)), d.conj())

        # 旋转数据以对齐坐标系
        stolt = torch.rot90(stolt, 2)

        return stolt

    def forward(self, sig):
        print("\nStarting WKA processing...")
        # 0. 检查输入信号
        print(f"Input signal shape: {sig.shape}")
        print(f"Input signal min: {torch.abs(sig).min().item()}")
        print(f"Input signal max: {torch.abs(sig).max().item()}")

        # 1. 二维FFT变换到频域
        s2df = self.fft2d_torch(sig)
        print(f"After FFT - min: {torch.abs(s2df).min().item()}, max: {torch.abs(s2df).max().item()}")

        # 2. 频域匹配滤波（距离压缩 + RCMC补偿）
        s2df_matched = self.matched_filtering_torch(s2df)
        print(
            f"After matched filter - min: {torch.abs(s2df_matched).min().item()}, max: {torch.abs(s2df_matched).max().item()}")

        # 3. Stolt插值（解决距离弯曲）
        s2df_stolt = self.stolt_interpolation_torch(s2df_matched)
        print(
            f"After Stolt interpolation - min: {torch.abs(s2df_stolt).min().item()}, max: {torch.abs(s2df_stolt).max().item()}")

        # 4. 二维IFFT到图像
        image = w1 * self.ifft2d_torch(s2df_stolt)
        print(f"After IFFT - min: {torch.abs(image).min().item()}, max: {torch.abs(image).max().item()}")

        return s2df, s2df_matched, s2df_stolt, image


if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    config = Config()
    print("Configuration:")
    print(f"  Range samples (Nrg): {config.Nrg}")
    print(f"  Azimuth samples (Naz): {config.Naz}")
    print(f"  Reference range (Rref): {config.Rref} m")
    print(f"  Pulse duration (Tr): {config.Tr} s")
    print(f"  Chirp rate (Kr): {config.Kr} Hz/s")

    # 创建信号生成器
    generator = SignalGenerator(config)

    # 创建点目标场景
    targets = [
        (19892.69596844, 1266.68940729),
        (19962.69596844, 1270.9707907),
        (20032.69596844, 1275.25217411)
    ]

    print("\nTarget positions:")
    for i, (x, y) in enumerate(targets):
        print(f"  Target {i + 1}: x={x:.2f}m, y={y:.2f}m")

    # 生成原始回拨信号
    print("\nGenerating SAR signal...")
    sar_signal = generator.generate_signal(targets)
    print(sar_signal)

    # 创建WKA处理器
    wka = WKA(config)

    # 处理信号得到图像
    print("\nProcessing SAR signal with ωK algorithm...")
    s2df, s2df_matched, s2df_stolt, image = wka.forward(sar_signal)

    sig_data = torch.abs(sar_signal).detach().cpu().numpy()
    filter_data = torch.abs(WKA.ifft2d_torch(s2df_matched)).detach().cpu().numpy()
    stolt_data = torch.abs(image).detach().cpu().numpy()

    plt.figure()
    plt.imshow(sig_data, aspect='auto', cmap='jet')
    plt.title('Original Echo Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.figure()
    plt.imshow(filter_data, aspect='auto', cmap='jet')
    plt.title('Matched Filtered Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.figure()
    plt.imshow(stolt_data, aspect='auto', cmap='jet')
    plt.title('Stolt Interpolated Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.show()