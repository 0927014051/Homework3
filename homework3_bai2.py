import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name, xsize):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=xsize * xsize)
        if data.size != xsize * xsize:
            raise Exception(f"Không thể đọc file {file_name}")
        return data.reshape(256, 256)
    except FileNotFoundError:
        raise Exception(f'Không thể mở file {file_name}')

def stretch(image):
    maxV = 255
    min_val = np.min(image)
    max_val = np.max(image)
    if min_val == max_val:
        return image
    stretched_factory = (maxV)/(max_val-min_val)
    stretched = (image - min_val)*stretched_factory
    return stretched.astype(np.uint8)

def plot_images_and_histograms(original, stretched):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image')

    plt.subplot(2, 2, 2)
    plt.imshow(stretched, cmap='gray')
    plt.title('Full-scale stretch')

    plt.subplot(2, 2, 3)
    plt.hist(original.ravel(), bins=256, range=(0, 256), color='b', alpha=0.7)
    plt.title('Histogram image')

    plt.subplot(2, 2, 4)
    plt.hist(stretched.ravel(), bins=256, range=(0, 256), color='g', alpha=0.7)
    plt.title('New histogram image')

    plt.show()

def main():
    xsize = 256
    x = read_file('dataset/ladybin.sec', xsize)
    y = stretch(x)
    plot_images_and_histograms(x, y)
if __name__ == '__main__':
    main()