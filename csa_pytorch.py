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

        # 创建距离时间轴
        tau = torch.linspace(
            2 * self.config.Rmin / self.c - self.config.Tr / 2,
            2 * self.config.Rmax / self.c + self.config.Tr / 2,
            self.config.Nr, dtype=torch.float64, device=self.device
        )

        # 创建距离频率轴 f_tau 和方位频率轴 f_eta
        f_eta_ref = 2 * self.Vr * torch.sin(self.theta_c) / self.lamda
        f_tau = torch.linspace(-config.Fr / 2, config.Fr / 2, config.Nr, dtype=torch.float64, device=device)
        f_eta = torch.linspace(-config.Fa / 2, config.Fa / 2, config.Na, dtype=torch.float64, device=device) + f_eta_ref

        # 扩展为矩阵用于向量化计算
        self.tau_mtx = tau.unsqueeze(0).repeat(self.config.Na, 1)  # [Na, Nr]
        self.f_tau = f_tau.repeat(config.Na, 1)  # [Na, Nr]
        self.f_eta = f_eta.reshape(-1, 1).repeat(1, config.Nr)  # [Na, Nr]

        # 参考多普勒中心频率下的徙动参数
        f_eta_ref_tensor = torch.tensor([f_eta_ref], dtype=torch.float64, device=device)
        self.D_fetaref_Vrref = torch.sqrt(1 - (self.c * f_eta_ref_tensor) ** 2 / (4 * self.Vr ** 2 * self.f0 ** 2))

        # 徙动参数
        self.D_feta_Vr = torch.sqrt(1 - (self.c * self.f_eta) ** 2 / (4 * self.Vr ** 2 * self.f0 ** 2))

        # 改变后的距离向调频率 K_m
        numerator = self.Kr * self.c * self.R_ref * self.f_eta ** 2
        denominator = 2 * self.Vr ** 2 * self.f0 ** 3 * self.D_feta_Vr ** 3
        self.Km = self.Kr / (1 - numerator / denominator)

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

    def azimuth_fft(self, sig):
        return self.fft1d_torch(sig, dim=0)

    def chirp_scaling(self, sig, tau):
        tau_prime = tau - 2 * self.R_ref / (self.c * self.D_feta_Vr)

        scaling_factor = self.D_fetaref_Vrref / self.D_feta_Vr - 1

        H_cs = torch.exp(1j * torch.pi * self.Km * scaling_factor * tau_prime ** 2)

        return sig * H_cs

    def range_fft(self, sig):
        return self.fft1d_torch(sig, dim=1)

    def range_processing(self, sig):
        H_rc = torch.exp(1j * torch.pi * self.D_feta_Vr / (self.Km * self.D_fetaref_Vrref) * self.f_tau ** 2)

        H_rcmc = torch.exp(
            1j * (4 * torch.pi * self.R_ref / self.c) * self.f_tau * (1 / self.D_feta_Vr - 1 / self.D_fetaref_Vrref))

        return sig * H_rc * H_rcmc

    def range_ifft(self, sig):
        return self.ifft1d_torch(sig, dim=1)

    def azimuth_compression(self, sig):
        D_ratio = self.D_feta_Vr / self.D_fetaref_Vrref
        R_term = (self.R_ref / self.D_feta_Vr - self.R_ref / self.D_fetaref_Vrref) ** 2

        H_p = torch.exp(-1j * (4 * torch.pi * self.Km / self.c ** 2) * (1 - D_ratio) * R_term)

        H_ac = torch.exp(1j * (4 * torch.pi * self.R_ref * self.f0 * self.D_feta_Vr / self.c))

        return sig * H_p * H_ac

    def azimuth_ifft(self, sig):
        return self.ifft1d_torch(sig, dim=0)

    def forward(self, sig):
        print("\nStarting optimized CSA processing...")
        start_time = time.time()

        if sig.device != self.device:
            sig = sig.to(self.device)

        # 1. 方位向FFT
        print("Performing Azimuth FFT...")
        S_eta_f = self.azimuth_fft(sig)
        print(f"Azimuth FFT completed in {time.time() - start_time:.2f}s")

        # 2. Chirp Scaling操作
        print("Performing Chirp Scaling...")
        S_eta_f_scaled = self.chirp_scaling(S_eta_f, self.tau_mtx)
        print(f"Chirp Scaling completed in {time.time() - start_time:.2f}s")

        # 3. 距离向FFT
        print("Performing Range FFT...")
        S_2df_scaled = self.range_fft(S_eta_f_scaled)
        print(f"Range FFT completed in {time.time() - start_time:.2f}s")

        # 4. 距离压缩+SRC+RCMC
        print("Performing Range Compression, SRC & RCMC...")
        S_2df_comp = self.range_processing(S_2df_scaled)
        print(f"Range Compression, SRC & RCMC completed in {time.time() - start_time:.2f}s")

        # 5. 距离向IFFT
        print("Performing Range IFFT...")
        S_eta_f_comp = self.range_ifft(S_2df_comp)
        print(f"Range IFFT completed in {time.time() - start_time:.2f}s")

        # 6. 方位向压缩
        print("Performing Azimuth Compression...")
        S_eta_f_ac = self.azimuth_compression(S_eta_f_comp)
        print(f"Azimuth Compression completed in {time.time() - start_time:.2f}s")

        # 7. 方位向IFFT
        print("Performing Azimuth IFFT...")
        S_image = self.azimuth_ifft(S_eta_f_ac)
        print(f"Total processing time: {time.time() - start_time:.2f}s")

        return S_2df_scaled, S_2df_comp, S_image


if __name__ == "__main__":
    config = Config()
    generator = SignalGenerator(config)

    targets = [
        (config.Xc, config.Yc),  # 场景中心
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
    sig_scaled_2df_np = torch.abs(sig_scaled_2df).detach().cpu().numpy()
    image_np = torch.abs(image).detach().cpu().numpy()

    # 可视化函数绘图
    visualize([sar_signal_np, sig_scaled_2df_np, image_np],
              ['Original Echo Signal', 'Chirp Scaled Signal', 'Final Processed Signal'])
