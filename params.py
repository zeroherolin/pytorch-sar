import math


class Config:
    def __init__(self):
        # 基础参数
        self.c = 3e8                                                   # 光速 (m/s)
        self.R0 = 30e3                                                 # 中心斜距 (m)
        self.H = 10e3                                                  # 飞行高度 (m)
        self.Vr = 250                                                  # 平台速度 (m/s)
        self.Tr = 10e-6                                                # 脉冲持续时间 (s)
        self.B = 100e6                                                 # 信号带宽 (Hz)
        self.Kr = self.B / self.Tr                                     # 距离向调频率 (Hz/s)
        self.f0 = 9.4e9                                                # 载波频率 (Hz)
        self.lamda = self.c / self.f0                                  # 波长 (m)
        self.Fr = 1.2 * self.B                                         # 距离向采样率 (Hz)
        self.Fa = 600                                                  # 方位向采样率 (Hz)
        self.Ka = 2 * self.Vr ** 2 / self.lamda / self.R0              # 方位向调频率 (Hz/s)
        self.D = 1                                                     # 天线真实孔径 (m)
        self.Ls = 0.886 * self.R0 * self.lamda / self.D                # 合成孔径长度 (m)

        # 场景参数
        self.Xc = math.sqrt(self.R0 ** 2 - self.H ** 2)                # 场景中心X坐标 (m)
        self.Yc = 0                                                    # 场景中心Y坐标 (m)
        self.Xo = 500                                                  # 场景半宽X (m)
        self.Yo = 300                                                  # 场景半宽Y (m)

        # 采样参数
        self.Rmin = math.sqrt(self.H ** 2 + (self.Xc - self.Xo) ** 2)  # 最近斜距 (m)
        self.Rmax = math.sqrt(self.H ** 2 + (self.Xc + self.Xo) ** 2)  # 最远斜距 (m)
        self.Ra = self.Ls + 2 * self.Yo                                # 方位向行走距离 (m)
        self.Na = None                                                 # 方位向采样点数 (将在信号生成时设置)
        self.Nr = None                                                 # 距离向采样点数 (将在信号生成时设置)

        # 参考参数
        self.theta_c = 0 /180 * math.pi                                # 波束中心斜视角 (rad)
        self.R_ref = self.R0                                           # 参考距离 (m)
