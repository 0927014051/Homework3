#Họ và tên: Trịnh Thanh Sơn
#MSSV: N20DCCN134
import numpy as np
import matplotlib.pyplot as plt
def read_file(file_name, x_size):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=x_size * x_size)
        if data.size != x_size * x_size:
            raise Exception(f'Read file {file_name} fail')
        return data.reshape(256, 256)
    except FileNotFoundError:
        print(f'ERROR: Can not open file {file_name}!')
def stretch(a):
    min = np.min(a)
    max = np.max(a)
    x = 255.0 / (max - min)
    str = np.round((a - min) * x)
    return str.astype(np.uint8)
def plot(data1, data2, data3, data4):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(data1, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image')

    plt.subplot(2, 2, 2)
    plt.imshow(data2, cmap='gray')
    plt.title('Full-scale stretch')

    plt.subplot(2, 2, 3)
    plt.bar(range(256), data3)
    plt.title('Histogram image')

    plt.subplot(2, 2, 4)
    plt.bar(range(256), data4)
    plt.title('New histogram image')
    plt.show()
def main():
    x_size = 256
    x = read_file('dataset/ladybin.sec', x_size)
    x_his = np.histogram(x, bins= np.arange(257), range=(0, 256))[0]
    y = stretch(x)
    y_his = np.histogram(y, bins= np.arange(257), range=(0, 256))[0]
    plot(x, y, x_his, y_his)
if __name__ == '__main__':
    main()







