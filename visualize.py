import matplotlib.pyplot as plt
import numpy as np


def visualize(imaga, title):
    for i, img in enumerate(imaga):
        plt.figure()
        plt.imshow(img, aspect='auto', cmap='jet', origin='upper')
        plt.title(title[i])
        plt.xlabel('Range Bins')
        plt.ylabel('Azimuth Bins')

    # 绘制局部
    max_val = np.max(imaga[-1])
    y, x = np.where(imaga[-1] == max_val)
    x0, y0 = x[0], y[0]
    sig1_crop = imaga[-1][y0 - 31:y0 + 32, x0 - 31:x0 + 32]
    plt.figure()
    plt.imshow(sig1_crop, aspect='auto', cmap='jet', origin='lower')
    plt.title('Local Contour of Matched Signal')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')

    plt.show()
