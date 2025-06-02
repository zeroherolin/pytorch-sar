import matplotlib.pyplot as plt
import numpy as np


def visualize(imaga, title):
    sar_signal_np, matched_np, stolt_np = imaga

    plt.figure()
    plt.imshow(sar_signal_np, aspect='auto', cmap='jet', origin='upper')
    plt.title(title[0])
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.figure()
    plt.imshow(matched_np, aspect='auto', cmap='jet', origin='lower')
    plt.title(title[1])
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.figure()
    plt.imshow(stolt_np, aspect='auto', cmap='jet', origin='lower')
    plt.title(title[2])
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    # 绘制局部
    max_val = np.max(stolt_np)
    y, x = np.where(stolt_np == max_val)
    x0, y0 = x[0], y[0]
    sig1_crop = stolt_np[y0 - 31:y0 + 32, x0 - 31:x0 + 32]
    plt.figure()
    plt.imshow(sig1_crop, aspect='auto', cmap='jet', origin='lower')
    plt.title('Local Contour of Matched Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.show()
