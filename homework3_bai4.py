
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name, size):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=size*size)
        if data.size != size*size:
            raise Exception(f"Không thể đọc file {file_name}")
        return data.reshape(size, size)
    except FileNotFoundError:
        raise Exception(f'Không thể mở file {file_name}')

def plot(data1, data2, data3, data4):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(data1, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(256), data2)
    plt.title('Original Histogram')
    plt.axis([0, 256, 0, 2500])
    plt.xticks(np.arange(0, 256, 16))
    plt.yticks(np.arange(0, 2500, 250))

    plt.subplot(2, 2, 3)
    plt.imshow(data3, cmap='gray', vmin=0, vmax=255)
    plt.title('Histogram Equalized Image')

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(256), data4)
    plt.title('Equalized Histogram')
    plt.axis([0, 256, 0, 2500])
    plt.xticks(np.arange(0, 256, 16))
    plt.yticks(np.arange(0, 2500, 250))

    plt.show()

def hist_equalization(image):
    equalized = cv2.equalizeHist(image)
    return equalized

def main():
    xsize = 256
    file_name = 'dataset/johnnybin.sec'
    x = read_file(file_name, xsize)
    if x is None:
        return
    x = x.reshape(xsize, xsize)
    hist_x = np.histogram(x, bins=256, range=(0, 256))[0]
    y = hist_equalization(x)
    hist_y = np.histogram(y, bins=256, range=(0, 256))[0]
    plot(x, hist_x, y, hist_y)

if __name__ == '__main__':
    main()