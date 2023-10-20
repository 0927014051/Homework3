#Họ và tên: Trịnh Thanh Sơn
#MSSV: N20DCCN134
import numpy as np
import matplotlib.pyplot as plt
def read_file(file_name, size):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=size*size)
    except FileNotFoundError:
        print(f'ERROR: Can not open file {file_name}!')
        return None
    return data
def plot(data1, data2, data3):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(data1, cmap='gray')
    plt.title('Original image')

    plt.subplot(2, 2, 2)
    plt.imshow(data2, cmap='gray', vmin=0, vmax=255)
    plt.title('J1 image')

    plt.subplot(2, 2, 3)
    plt.imshow(data3, cmap='gray', vmin=0, vmax=255)
    plt.title('J2 image')

    plt.show()

def compute_match_measure(image, template):
    J1 = np.zeros_like(image, dtype=int)
    tmpRows, tmpCols = template.shape[0]//2, template.shape[1]//2
    for row in range(tmpRows, image.shape[0] - tmpRows):
        for col in range(tmpCols, image.shape[1] - tmpCols):
            window_set = image[row - tmpRows:row + tmpRows + 1, col - tmpCols:col + tmpCols + 1]
            J1[row, col] = np.sum(window_set == template)
    
    return J1
def bult_templ():
    tmpRows = 47
    tmpCols = 15
    tmp = np.zeros((tmpRows, tmpCols), dtype=np.uint8)
    for row in range(10):
        for col in range(tmpCols):
            tmp[row, col] = 0
    for row in range(10, 16):
        for col in range(tmpCols):
            tmp[row, col] = 255
    for row in range(17, 37):
        for col in range(6):
            tmp[row, col] = 0
        for col in range(6, 10):
            tmp[row, col] = 255
        for col in range(10, tmpCols):
            tmp[row, col] = 0
    for row in range(38, tmpRows):
        for col in range(tmpCols):
            tmp[row, col] = 0
    return tmp
def stretch(a):
    min = np.min(a)
    max = np.max(a)
    x = 255.0 / (max - min)
    str = np.round((a - max) * x)
    return str.astype(np.uint8)
def main():
    x_size = 256
    file_name = 'dataset/actontBinbin.sec'
    x = read_file(file_name, x_size)
    if x is None:
        return
    x = x.reshape(x_size, x_size)
    tmp = bult_templ()
    J1 = compute_match_measure(x, tmp)
    J1 = stretch(J1)
    count = np.sort(J1.ravel())[::-1]
    threshold = count[1]
    J2 = 255 * (J1 >= threshold)
    plot(x, J1, J2)
if __name__ == '__main__':
    main()