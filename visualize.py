import matplotlib.pyplot as plt
import numpy as np


def visualize(imaga):
    sar_signal_np, matched_np, stolt_np = imaga

    # 绘制原始回波信号
    plt.figure()
    plt.imshow(sar_signal_np, aspect='auto', cmap='jet', origin='upper')
    plt.title('Original Echo Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    # 绘制匹配滤波后的信号
    plt.figure()
    plt.imshow(matched_np, aspect='auto', cmap='jet', origin='lower')
    plt.title('Matched Filtered Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    # 绘制Stolt插值后的信号
    plt.figure()
    plt.imshow(stolt_np, aspect='auto', cmap='jet', origin='lower')
    plt.title('Stolt Interpolated Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    # 绘制局部
    max_val = np.max(stolt_np)
    y, x = np.where(stolt_np == max_val)
    x0, y0 = x[0], y[0]
    sig1_crop = stolt_np[y0 - 15:y0 + 16, x0 - 15:x0 + 16]
    plt.figure()
    plt.imshow(sig1_crop, aspect='auto', cmap='jet', origin='lower')
    plt.title('Local Contour of Matched Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.show()
