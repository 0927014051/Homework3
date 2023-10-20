#Họ và tên: Trịnh Thanh Sơn
#MSSV: N20DCCN134
import numpy as np
import matplotlib.pyplot as plt
def read_file(file_name, size):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=size*size)
    except FileNotFoundError:
        print(f'\nERROR: Can not open file {file_name}!')
        return None
    return data
def plot(data1, data2, data3, data4):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(data1, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image')

    plt.subplot(2, 2, 2)
    plt.imshow(data3, cmap='gray', vmin=0, vmax=255)
    plt.title('Histogram equalized image')

    plt.subplot(2, 2, 3)
    plt.bar(np.arange(256), data2)
    plt.title('Original histogram')
    plt.axis([0, 256, 0, 2500])
    plt.xticks(np.arange(0, 256, 16))
    plt.yticks(np.arange(0, 2500, 250))

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(256), data4)
    plt.title('Equalized histogram')
    plt.axis([0, 256, 0, 2500])
    plt.xticks(np.arange(0, 256, 16))
    plt.yticks(np.arange(0, 2500, 250))

    plt.show()
def stretch(a):
    min = np.min(a)
    max = np.max(a)
    x = 255.0 / (max - min)
    stretched = np.round((a - min) * x)
    return stretched.astype(np.uint8)
def hist_equalization(image):
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    num_pixels = image.size
    x = hist/num_pixels
    y = np.cumsum(x)
    J = y[image]
    K = stretch(J)
    return K
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