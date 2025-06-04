import torch
import torch.nn as nn
import time

from params import Config
from sig_gen import SignalGenerator
from visualize import visualize


class RDA(nn.Module):
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        print(f"Initializing optimized RDA processor on {device}...")

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

        # 预计算距离压缩滤波器
        self.range_comp_filter = torch.exp(1j * torch.pi * self.f_tau ** 2 / self.Kr)

        # 预计算方位压缩滤波器
        D_feta = torch.sqrt(1 - (self.lamda * self.f_eta) ** 2 / (4 * self.Vr ** 2))
        self.azimuth_comp_filter = torch.exp(1j * 4 * torch.pi * self.R_ref * D_feta / self.lamda)

        # 预计算RCMC参数
        self.RCMC_shift = (self.R_ref * (1 - D_feta) * 4 / self.c) * self.Fr

        # 为RCMC插值创建索引网格
        self.n = torch.arange(config.Nr, dtype=torch.float64, device=device)
        self.m = torch.arange(config.Na, dtype=torch.float64, device=device)

        # 创建用于插值的sinc滤波器组
        self.sinc_filter = self.create_sinc_filter(6) # 使用6点sinc插值

    def create_sinc_filter(self, kernel_size):
        offsets = torch.arange(-kernel_size // 2, kernel_size // 2, dtype=torch.float64, device=self.device)
        sinc_filter = torch.sinc(offsets)
        return sinc_filter / sinc_filter.sum() # 归一化

    @staticmethod
    def fft1d_torch(x, dim=-1):
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.fft(x, dim=dim)
        x = torch.fft.fftshift(x, dim=dim)
        return x

    @staticmethod
    def ifft1d_torch(x, dim=-1):
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.ifft(x, dim=dim)
        x = torch.fft.fftshift(x, dim=dim)
        return x

    def range_fft(self, sig):
        return self.fft1d_torch(sig, dim=1)

    def range_compression(self, sig):
        return sig * self.range_comp_filter

    def range_ifft(self, sig):
        return self.ifft1d_torch(sig, dim=1)

    def azimuth_fft(self, sig):
        return self.fft1d_torch(sig, dim=0)

    def rcmc_interpolation(self, sig_rd):
        Na, Nr = sig_rd.shape
        kernel_size = len(self.sinc_filter)

        # 计算需要的插值位置
        shift_per_row = self.RCMC_shift[:, 0] # 每行的偏移量 [Na]

        # 创建插值位置网格 (向量化)
        n_grid = self.n.unsqueeze(0) - shift_per_row.unsqueeze(1) # [Na, Nr]

        # 计算整数和小数部分
        n_int = torch.floor(n_grid).long()
        n_frac = n_grid - n_int

        # 为边界处理创建有效掩码
        valid_mask = (n_int >= 0) & (n_int < Nr - kernel_size + 1)

        # 初始化输出
        sig_rcmc = torch.zeros_like(sig_rd, dtype=torch.complex128, device=self.device)

        # 使用卷积方法进行高效插值
        for k in range(kernel_size):
            # 计算当前核位置的索引
            idx = n_int + k

            # 创建边界安全的索引
            idx_clamped = torch.clamp(idx, 0, Nr - 1)

            # 收集样本值
            samples = torch.gather(sig_rd, 1, idx_clamped)

            # 应用权重 (使用预计算的sinc滤波器)
            weight = self.sinc_filter[k]

            # 只对有效位置累加
            sig_rcmc += torch.where(valid_mask, samples * weight, 0)

        return sig_rcmc

    def azimuth_compression(self, sig):
        return sig * self.azimuth_comp_filter

    def azimuth_ifft(self, sig):
        return self.ifft1d_torch(sig, dim=0)

    def forward(self, sig):
        print("\nStarting optimized RDA processing...")
        start_time = time.time()

        if sig.device != self.device:
            sig = sig.to(self.device)

        # 1. 距离向FFT
        print("Performing Range FFT...")
        sig_range_fft = self.range_fft(sig)
        print(f"Range FFT completed in {time.time() - start_time:.2f}s")

        # 2. 距离压缩（相位匹配）
        print("Performing Range Compression...")
        sig_range_comp = self.range_compression(sig_range_fft)
        print(f"Range Compression completed in {time.time() - start_time:.2f}s")

        # 3. 距离向IFFT
        print("Performing Range IFFT...")
        sig_range_ifft = self.range_ifft(sig_range_comp)
        print(f"Range IFFT completed in {time.time() - start_time:.2f}s")

        # 4. 方位向FFT（到距离多普勒域）
        print("Performing Azimuth FFT...")
        sig_azimuth_fft = self.azimuth_fft(sig_range_ifft)
        print(f"Azimuth FFT completed in {time.time() - start_time:.2f}s")

        # 5. 距离徙动校正（RCMC）
        print("Performing Range Cell Migration Correction (RCMC)...")
        sig_rcmc = self.rcmc_interpolation(sig_azimuth_fft)
        print(f"RCMC completed in {time.time() - start_time:.2f}s")

        # 6. 方位压缩（相位匹配）
        print("Performing Azimuth Compression...")
        sig_azimuth_comp = self.azimuth_compression(sig_rcmc)
        print(f"Azimuth Compression completed in {time.time() - start_time:.2f}s")

        # 7. 方位向IFFT（到图像域）
        print("Performing Azimuth IFFT...")
        image = self.azimuth_ifft(sig_azimuth_comp)
        print(f"Total processing time: {time.time() - start_time:.2f}s")

        return sig_range_comp, sig_rcmc, image


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
    rda = RDA(config)
    sig_range_comp, sig_rcmc, image = rda.forward(sar_signal)

    # 转换结果为NumPy数组
    sar_signal_np = torch.abs(sar_signal).detach().cpu().numpy()
    sig_rcmc_np = torch.abs(rda.range_ifft(sig_rcmc)).detach().cpu().numpy()
    image_np = torch.abs(image).detach().cpu().numpy()

    # 可视化函数绘图
    visualize([sar_signal_np, sig_rcmc_np, image_np],
              ['Original Echo Signal', 'RCMC Signal', 'Final Processed Image'])
