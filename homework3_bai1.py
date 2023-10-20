#Họ và tên: Trịnh Thanh Sơn
#MSSV: N20DCCN34
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
def read_file(file_name, size):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=size*size)
    except FileNotFoundError:
        print(f'ERROR: Can not open file {file_name}!')
        return None
    return data
def plot_a(data1, data2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(data1, cmap='gray', norm=NoNorm())
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(data2, cmap='gray', norm=NoNorm())
    plt.title('Thresholding result')
    plt.axis('image')
def plot_b(data, file_name):
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.imshow(data, cmap='gray', norm=NoNorm())
    plt.axis('image')
    plt.title('CAU B', fontsize =12)
    plt.savefig(file_name, format='eps')
    plt.savefig(file_name[:-4]+'.pdf', format='pdf')
def main():
    #CAU A
    x_size = 256
    thresh = 95
    file_name = 'dataset/Mammogrambin.sec'
    x_original = read_file(file_name, x_size)
    if x_original is None:
        return
    x_original = x_original.reshape(x_size, x_size)
    x = 255 * (x_original >= thresh)
    plot_a(x_original, x)
    #CAU B
    y = np.zeros((x_size, x_size), dtype=np.uint8)
    for row in range(x_size):
        for col in range(x_size):
            if x[row, col] == 255:
                y[row, col] = 0
            else:
                if((col > 0 and x[row, col-1] == 255) or
                    (col < x_size - 1 and x[row, col + 1] == 255) or
                    (row > 0 and x[row-1, col] == 255) or
                    (row < x_size -1 and x[row + 1, col] == 255)):
                    y[row, col] = 255
                else:
                    y[row, col] = 0
    plot_b(y, 'Approximate contour image generation')
    plt.show()
if __name__ == "__main__":
    main()