import torch
import torch.nn as nn
import time

from params import Config
from sig_gen import SignalGenerator
from visualize import visualize


class CSA(nn.Module):
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        print(f"Initializing optimized CSA processor on {device}...")

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

        # 预计算徙动参数
        self.D_feta_Vr = torch.sqrt(1 - (self.c * self.f_eta) ** 2 / (4 * self.Vr ** 2 * self.f0 ** 2))

        # 预计算改变后的距离向调频率
        numerator = self.Kr * self.c * self.R_ref * self.f_eta ** 2
        denominator = 2 * self.Vr ** 2 * self.f0 ** 3 * self.D_feta_Vr ** 3
        self.Km = self.Kr / (1 - numerator / denominator)

        # 预计算参考徙动参数
        f_eta_ref_tensor = torch.tensor([f_eta_ref], dtype=torch.float64, device=device)
        self.D_fetaref_Vrref = torch.sqrt(1 - (self.c * f_eta_ref_tensor) ** 2 / (4 * self.Vr ** 2 * self.f0 ** 2))

        # 预计算变标方程参数
        self.scaling_factor = self.D_fetaref_Vrref / self.D_feta_Vr - 1

        # 预计算距离压缩相位因子
        self.range_comp_phase = torch.exp(1j * torch.pi * self.D_feta_Vr /
                                          (self.D_fetaref_Vrref * self.Km) * self.f_tau ** 2)

        # 预计算RCMC相位因子
        self.rcmc_phase = torch.exp(1j * 4 * torch.pi / self.c *
                                    (1 / self.D_feta_Vr - 1 / self.D_fetaref_Vrref) *
                                    self.R_ref * self.f_tau)

        # 预计算方位压缩相位因子
        self.azimuth_comp_phase = torch.exp(1j * 4 * torch.pi * self.R_ref * self.f0 * self.D_feta_Vr / self.c)

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

    def azimuth_fft(self, sig):
        return self.fft1d_torch(sig, dim=0)

    def chirp_scaling(self, sig, tau):
        # 计算新的距离时间
        tao_pie = tau - 2 * self.R_ref / (self.c * self.D_feta_Vr)

        # 计算变标方程
        scaling_phase = torch.exp(1j * torch.pi * self.Km * self.scaling_factor * tao_pie ** 2)

        return sig * scaling_phase

    def range_fft(self, sig):
        return self.fft1d_torch(sig, dim=1)

    def range_compression(self, sig):
        # 应用距离压缩和RCMC相位因子
        return sig * self.range_comp_phase * self.rcmc_phase

    def range_ifft(self, sig):
        return self.ifft1d_torch(sig, dim=1)

    def azimuth_compression(self, sig):
        # 应用方位压缩相位因子
        return sig * self.azimuth_comp_phase

    def azimuth_ifft(self, sig):
        return self.ifft1d_torch(sig, dim=0)

    def forward(self, sig):
        print("\nStarting optimized CSA processing...")
        start_time = time.time()

        if sig.device != self.device:
            sig = sig.to(self.device)

        # 0. 准备距离时间轴
        tau = torch.linspace(2 * self.config.Rmin / self.c - self.config.Tr / 2,
                             2 * self.config.Rmax / self.c + self.config.Tr / 2,
                             self.config.Nr, dtype=torch.float64, device=self.device)
        tau_mtx = tau.unsqueeze(0).repeat(self.config.Na, 1) # [Na, Nr]

        # 1. 方位向FFT (到距离多普勒域)
        print("Performing Azimuth FFT...")
        sig_rd = self.azimuth_fft(sig)
        print(f"Azimuth FFT completed in {time.time() - start_time:.2f}s")

        # 2. Chirp Scaling操作
        print("Performing Chirp Scaling...")
        sig_scaled = self.chirp_scaling(sig_rd, tau_mtx)
        print(f"Chirp Scaling completed in {time.time() - start_time:.2f}s")

        # 3. 距离向FFT (到二维频域)
        print("Performing Range FFT...")
        sig_scaled_2df = self.range_fft(sig_scaled)
        print(f"Range FFT completed in {time.time() - start_time:.2f}s")

        # 4. 距离压缩、SRC和一致RCMC
        print("Performing Range Compression, SRC & RCMC...")
        sig_rg_compressed = self.range_compression(sig_scaled_2df)
        print(f"Range Compression, SRC & RCMC completed in {time.time() - start_time:.2f}s")

        # 5. 距离向IFFT (回距离多普勒域)
        print("Performing Range IFFT...")
        sig_rd_compressed = self.range_ifft(sig_rg_compressed)
        print(f"Range IFFT completed in {time.time() - start_time:.2f}s")

        # 6. 方位向压缩
        print("Performing Azimuth Compression...")
        sig_az_compressed = self.azimuth_compression(sig_rd_compressed)
        print(f"Azimuth Compression completed in {time.time() - start_time:.2f}s")

        # 7. 方位向IFFT (到图像域)
        print("Performing Azimuth IFFT...")
        image = self.azimuth_ifft(sig_az_compressed)
        print(f"Total processing time: {time.time() - start_time:.2f}s")

        return sig_scaled_2df, sig_rg_compressed, image


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
    csa = CSA(config)
    sig_scaled_2df, sig_rg_compressed, image = csa.forward(sar_signal)

    # 转换结果为NumPy数组
    sar_signal_np = torch.abs(sar_signal).detach().cpu().numpy()
    sig_scaled_2df_np = torch.abs(CSA.ifft2d_torch(sig_scaled_2df)).detach().cpu().numpy()
    image_np = torch.abs(image).detach().cpu().numpy()

    # 可视化函数绘图
    visualize([sar_signal_np, sig_scaled_2df_np, image_np],
              ['Original Echo Signal', 'Chirp Scaled Signal', 'Final Processed Signal'])