import math


class Config:
    def __init__(self):
        self.c = 3e8                                                  # 光速 (m/s)
        self.Rc = 20e3                                                # 中心斜距 (m)
        self.Vr = 150                                                 # 平台速度 (m/s)
        self.Tr = 2.5e-6                                              # 脉冲持续时间 (s)
        self.Kr = 20e12                                               # 距离向调频率 (Hz/s)
        self.fc = 5.3e9                                               # 载波频率 (Hz)
        self.lamda = self.c / self.fc                                 # 波长 (m)
        self.Fr = 60e6                                                # 距离向采样率 (Hz)
        self.Fa = 100                                                 # 方位向采样率 (Hz)
        self.Naz = 256                                                # 方位向采样点数
        self.Nrg = 320                                                # 距离向采样点数
        self.theta_c = 3.5 / 180 * math.pi                            # 波束中心斜视角 (rad)
        self.Rref = self.Rc * math.cos(self.theta_c)                  # 参考距离 (m)
        self.fac = 2 * self.Vr * math.sin(self.theta_c) / self.lamda  # 多普勒中心频率 (Hz)
