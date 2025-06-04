# Pytorch-based SAR Imaging

<img src="assets/flow.png" width=900/>

## ωKA (Wavenumber Domain Algorithm)

- [wka_pytorch](wka_pytorch.py)

### 1 二维FFT

原始信号通过二维$\text{FFT}$变换到二维频域：

$$S_{\text{2df}}(f_\tau, f_\eta) = \text{FFT2}\left(\text{Sig}(\tau, \eta)\right)$$

- 变量：距离时间$\tau$，方位时间$\eta$，距离频率$f_\tau$，方位频率$f_\eta$
```
f_η (方位频率)
    ↑
    |
    ·------→ f_τ (距离频率)
```

### 2 匹配滤波

参考函数（补偿参考距离处相位历程）：

$$H_{\text{ref}}(f_\tau, f_\eta) = \exp\left\{j\frac{4\pi R_{\text{ref}}}{c}\sqrt{(f_0 + f_\tau)^2 - \left(\frac{c f_\eta}{2V_{r_{\text{ref}}}}\right)^2} + j\pi\frac{f_\tau^2}{K_r}\right\}  \tag{8.3}$$

匹配滤波输出：

$$S_{\text{2df,matched}}(f_\tau, f_\eta) = H_{\text{ref}}(f_\tau, f_\eta) \cdot S_{\text{2df}}(f_\tau, f_\eta)$$

### 3 Stolt插值

频率映射：

$$f_\tau' = \sqrt{(f_0 + f_\tau)^2 - \left(\frac{c f_\eta}{2V_r}\right)^2} - f_0  \tag{8.5}$$

沿距离频率轴插值：

$$S_{\text{2df,stolt}}(f_\tau', f_\eta) = \text{interp}_{f_\tau} \left[ S_{\text{2df,matched}}(f_\tau, f_\eta) \to f_\tau' \right]$$

例如通过$\text{sinc}$插值重采样：

$$S_{\text{2df,stolt}}(f_\tau', f_\eta) = \sum_{n} S_{\text{2df,matched}}(f_{\tau,n}, f_\eta) \cdot \text{sinc}\left(\frac{f_\tau' - f_{\tau,n}}{\Delta f_\tau}\right)$$

- 参数：距离频率采样间隔$\Delta f_\tau = \frac{F_r}{N_r}$，第n个距离频率点$f_{\tau,n} = -\frac{F_r}{2} + n \cdot \Delta f_\tau$

### 4 参考距离相位校正

相位补偿：

$$\Phi_{\text{comp}} = \exp\left(-j\frac{4\pi R_{\text{ref}}}{c}f_\tau'\right)$$

相位补偿输出：

$$S_{\text{2df,comp}}(f_\tau, f_\eta) = \Phi_{\text{comp}} \cdot S_{\text{2df,stolt}}(f_\tau', f_\eta)$$

### 5 二维IFFT

最终图像生成：

$$S_{\text{image}}(\tau, \eta) = \text{IFFT2}\left(S_{\text{2df,comp}}(f_\tau, f_\eta)\right)$$

## CSA (Chirp Scaling Algorithm)

- [csa_pytorch](csa_pytorch.py)

### 1 方位向FFT

原始信号通过方位向$\text{FFT}$变换到距离多普勒域：

$$S_{\eta f}(\tau, f_\eta) = \text{FFT}_\eta\left(\text{Sig}(\tau, \eta)\right)$$

### 2 线性调频缩放（Chirp Scaling）

徙动参数：

$$D(f_\eta, V_{r}) = \sqrt{1 - \frac{(c f_\eta)^2}{4 (V_r f_0)^2}} \tag{7.17}$$

新的距离时间：

$$\tau' = \tau - \frac{2 R_{\text{ref}}}{c D(f_\eta, V_{r_{\text{ref}}})} \tag{7.27}$$

变标方程：

$$H_{CS}(\tau', f_\eta) = \exp\left\{j \pi K_m \left[ \frac {D(f_{\eta_{\text{ref}}}, V_{r_{\text{ref}}})} {D(f_\eta, V_{r_{\text{ref}}})} - 1 \right] (\tau')^2 \right\} \tag{7.30}$$

变标输出：

$$S_{\eta f,\text{scaled}}(\tau, f_\eta) = H_{CS}(\tau', f_\eta) \cdot S_{\eta f}(\tau, f_\eta)$$

- 参数：多普勒中心频率$f_{\eta_{\text{ref}}} = \frac {2 V_{r_{\text{ref}}}} {\lambda} \sin{\theta_{s,\text{ref}}}$，改变后的距离向调频率
$K_m(f_\eta) = \frac{K_r}{1 - \dfrac{K_r c R_{\text{ref}} f_\eta^2}{2 V_r^2 f_0^3 \left[ D(f_\eta, V_r) \right]^3}}$

### 3 距离向FFT

变换到二维频域：

$$S_{\text{2df,scaled}}(f_\tau, f_\eta) = \text{FFT}_\tau\left(S_{\eta f,\text{scaled}}(\tau, f_\eta)\right)$$

### 4 距离压缩（RC）、二次距离压缩（SRC）和一致距离徙动校正（RCMC）

距离压缩（RC+SRC）：

$$H_{\text{rc}}(f_\tau, f_\eta) = \exp\left\{ j\pi \frac{D(f_\eta, V_r)}{K_m D(f_{\eta_{\text{ref}}}, V_r)} f_\tau^2 \right\} \tag{7.32}$$

距离徙动校正（RCMC）：

$$H_{\text{rcmc}}(f_\tau, f_\eta) = \exp\left\{ j \frac{4\pi R_{\text{ref}}}{c} f_\tau \left[ \frac{1}{D(f_\eta, V_{r_{\text{ref}}})} - \frac{1}{D(f_{\eta_{\text{ref}}}, V_{r_{\text{ref}}})} \right] \right\} \tag{7.32}$$

补偿后输出：
$$S_{\text{2df,comp}}(f_\tau, f_\eta) = H_{\text{rc}}(f_\tau, f_\eta) \cdot H_{\text{rcmc}}(f_\tau, f_\eta) \cdot S_{\text{2df,scaled}}(f_\tau, f_\eta)$$

### 5 距离向IFFT

变换回距离多普勒域：

$$S_{\eta f,\text{comp}}(\tau, f_\eta) = \text{IFFT}_\tau\left( S_{\text{2df,comp}}(f_\tau, f_\eta) \right)$$

### 6 方位压缩

残余相位补偿：

$$H_p(\tau, f_\eta) = \exp \left\{ -j \frac{4\pi K_m}{c^2} \left[ 1 - \frac{D(f_\eta, V_{r_{\text{ref}}})}{D(f_{\eta_{\text{ref}}}, V_{r_{\text{ref}}})} \right] \left[ \frac{R_0}{D(f_\eta, V_r)} - \frac{R_{\text{ref}}}{D(f_{\eta_{\text{ref}}}, V_r)} \right]^2 \right\} \tag{7.32}$$

方位压缩滤波器：

$$H_{\text{ac}}(f_\eta) = \exp\left\{ j \frac{4 \pi R_0 f_0 D(f_\eta, V_r)} {c} \right\} \tag{7.32}$$

压缩和补偿后输出：

$$S_{\eta f,\text{ac}}(\tau, f_\eta) = H_p(\tau, f_\eta) \cdot H_{\text{ac}}(f_\eta) \cdot S_{\eta f,\text{comp}}(\tau, f_\eta)$$

### 7 方位向IFFT

变换回时域得到最终图像：

$$S_{\text{image}}(\tau, \eta) = \text{IFFT}_\eta\left( S_{\eta f,\text{ac}}(\tau, f_\eta) \right)$$

## RDA (Range Doppler Algorithm)

- [rda_pytorch](rda_pytorch.py)
