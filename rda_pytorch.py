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
        self.R0 = torch.tensor(config.R0, dtype=torch.float64, device=self.device)
        self.c = torch.tensor(config.c, dtype=torch.float64, device=self.device)
        self.Kr = torch.tensor(config.Kr, dtype=torch.float64, device=self.device)

        # 创建距离时间轴
        tau = torch.linspace(
            2 * self.config.Rmin / self.c - self.config.Tr / 2,
            2 * self.config.Rmax / self.c + self.config.Tr / 2,
            self.config.Nr, dtype=torch.float64, device=self.device
        )  # [Nr]
        self.tau = tau.reshape(1, -1)  # [1, Nr]

        # 创建距离频率轴 f_tau 和方位频率轴 f_eta
        f_eta_ref = 2 * self.Vr * torch.sin(self.theta_c) / self.lamda
        f_tau = torch.linspace(-config.Fr / 2, config.Fr / 2, config.Nr, dtype=torch.float64, device=device)  # [Nr]
        f_eta = torch.linspace(-config.Fa / 2, config.Fa / 2, config.Na, dtype=torch.float64, device=device) + f_eta_ref  # [Na]

        # 扩展为矩阵用于向量化计算
        self.f_tau = f_tau.repeat(config.Na, 1)  # [Na, Nr]
        self.f_eta = f_eta.reshape(-1, 1).repeat(1, config.Nr)  # [Na, Nr]

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
        return self.fft1d_torch(sig, dim=1)  # [Na, Nr]

    def range_compression(self, S_rf):
        H_rc = torch.exp(1j * torch.pi * self.f_tau ** 2 / self.Kr)
        return S_rf * H_rc  # [Na, Nr]

    def range_ifft(self, S_rf_comp):
        return self.ifft1d_torch(S_rf_comp, dim=1)  # [Na, Nr]

    def azimuth_fft(self, S_rc):
        return self.fft1d_torch(S_rc, dim=0)  # [Na, Nr]

    def rcmc_interpolation(self, S_rd):
        Interp_Pi = 6  # 插值核宽度

        # 获取方位频率向量
        f_eta_vec = self.f_eta[:, 0]  # [Na]

        # 计算时间徙动量
        delta_tau = (self.lamda ** 2 * self.R0 * f_eta_vec ** 2) / (4 * self.c * self.Vr ** 2)  # [Na]

        # 将时间徙动转换为距离采样点偏移量
        delta_n = delta_tau * self.Fr  # [Na]

        # 创建输出张量
        S_rd_rcmc = torch.zeros_like(S_rd)  # [Na, Nr]

        # 分批次处理
        batch_size = 32  # 根据GPU内存调整

        for i_start in range(0, config.Na, batch_size):
            i_end = min(i_start + batch_size, config.Na)
            batch_slice = slice(i_start, i_end)
            batch_len = i_end - i_start

            # 获取当前批次的数据
            S_rd_batch = S_rd[batch_slice, :]  # [batch, Nr]
            delta_n_batch = delta_n[batch_slice]  # [batch]

            # 创建距离索引
            j_index = torch.arange(config.Nr, device=self.device, dtype=torch.float64).unsqueeze(0)  # [1, Nr]
            j_index = j_index.expand(batch_len, -1)  # [batch, Nr]

            # 计算插值位置
            interp_pos = j_index + delta_n_batch.unsqueeze(1)  # [batch, Nr]

            # 计算整数部分和小数部分
            n0 = torch.floor(interp_pos).long()  # [batch, Nr]
            u = interp_pos - n0.float()  # [batch, Nr]

            # 处理边界：越界索引设为边界值
            n0 = n0.clamp(0, config.Nr - 1)

            # 创建插值核偏移
            offsets = torch.arange(-Interp_Pi // 2, Interp_Pi // 2, device=self.device, dtype=torch.long)  # [Pi]

            # 收集所有插值点索引
            indices = n0.unsqueeze(2) + offsets  # [batch, Nr, Pi]

            # 处理边界：越界索引设为边界值
            indices = indices.clamp(0, config.Nr - 1)

            # 收集样本值
            batch_idx = torch.arange(batch_len, device=self.device).view(batch_len, 1, 1)  # [batch, 1, 1]
            samples = S_rd_batch[batch_idx, indices]  # [batch, Nr, Pi]

            # 计算sinc权重
            diff = u.unsqueeze(2) - offsets.float()  # [batch, Nr, Pi]
            weights = torch.sinc(diff)  # [batch, Nr, Pi]

            # 应用权重并求和
            interp_result = (samples * weights).sum(dim=2)  # [batch, Nr]

            # 存储结果
            S_rd_rcmc[batch_slice, :] = interp_result  # [Na, Nr]

        return S_rd_rcmc  # [Na, Nr]

    def azimuth_compression(self, S_rd_rcmc):
        D_feta = torch.sqrt(1 - (self.lamda * self.f_eta) ** 2 / (4 * self.Vr ** 2))  # [Na, Nr]
        H_ac = torch.exp(1j * 4 * torch.pi * self.R0 * D_feta / self.lamda)  # [Na, Nr]
        return S_rd_rcmc * H_ac  # [Na, Nr]

    def azimuth_ifft(self, S_rd_ac):
        return self.ifft1d_torch(S_rd_ac, dim=0)  # [Na, Nr]

    def forward(self, sig):
        print("\nStarting optimized RDA processing...")
        start_time = time.time()

        if sig.device != self.device:
            sig = sig.to(self.device)

        # 1. 距离向FFT
        print("Performing Range FFT...")
        S_rf = self.range_fft(sig)
        print(f"Range FFT completed at {time.time() - start_time:.6f}s")

        # 2. 距离压缩
        print("Performing Range Compression...")
        S_rf_comp = self.range_compression(S_rf)
        print(f"Range Compression completed at {time.time() - start_time:.6f}s")

        # 3. 距离向IFFT
        print("Performing Range IFFT...")
        S_rc = self.range_ifft(S_rf_comp)
        print(f"Range IFFT completed at {time.time() - start_time:.6f}s")

        # 4. 方位向FFT
        print("Performing Azimuth FFT...")
        S_rd = self.azimuth_fft(S_rc)
        print(f"Azimuth FFT completed at {time.time() - start_time:.6f}s")

        # 5. 距离徙动校正 (RCMC)
        print("Performing Range Cell Migration Correction (RCMC)...")
        S_rd_rcmc = self.rcmc_interpolation(S_rd)
        print(f"RCMC completed at {time.time() - start_time:.6f}s")

        # 6. 方位压缩
        print("Performing Azimuth Compression...")
        S_rd_ac = self.azimuth_compression(S_rd_rcmc)
        print(f"Azimuth Compression completed at {time.time() - start_time:.6f}s")

        # 7. 方位向IFFT
        print("Performing Azimuth IFFT...")
        S_image = self.azimuth_ifft(S_rd_ac)
        print(f"Total processing time: {time.time() - start_time:.6f}s")

        return S_rf_comp, S_rd_rcmc, S_image


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
    print(f"Signal generation completed in {gen_time:.6f} seconds")

    # 处理信号
    rda = RDA(config)
    srf_comp, srd_rcmc, image = rda.forward(sar_signal)

    # 转换结果为NumPy数组
    sar_signal_np = torch.abs(sar_signal).detach().cpu().numpy()
    srf_comp_np = torch.abs(rda.range_ifft(srf_comp)).detach().cpu().numpy()
    srd_rcmc_np = torch.abs(rda.azimuth_ifft(srd_rcmc)).detach().cpu().numpy()
    image_np = torch.abs(image).detach().cpu().numpy()

    # 可视化函数绘图
    visualize([sar_signal_np, srf_comp_np, image_np],
              ['Original Echo Signal', 'After Range Compressed', 'After RCMC',
               'Final Processed Image\n(After Azimuth Compression)'])
