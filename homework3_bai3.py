import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name, size):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=size * size)
        if data.size != size * size:
            raise Exception(f"Không thể đọc file {file_name}")
        return data.reshape(size, size)
    except FileNotFoundError:
        raise Exception(f'Không thể mở file {file_name}')


def build_template():
    template = np.zeros((47, 15), dtype=np.uint8)
    template[10:16, :] = 255
    template[17:37, 6:10] = 255
    return template


def compute_match_measure(image, template):
    J1 = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    J1 = (J1 * 255).astype(np.uint8)
    return J1


def threshold_image(image, threshold):
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresholded

def stretch(a):
    min_val = np.min(a)
    max_val = np.max(a)
    x = 255.0 / (max_val - min_val)
    stretched = np.round((a - min_val) * x)
    return stretched.astype(np.uint8)
def main():
    xsize = 256
    file_name = 'dataset/actontBinbin.sec'
    x = read_file(file_name, xsize)
    template = build_template()

    J1 = compute_match_measure(x, template)
    J1 = stretch(J1)
    count = np.sort(J1.ravel())[::-1]
    threshold = count[1]
    J2 = 255 * (J1 >= threshold)
    plot_images(x, J1, J2)


def plot_images(data1, data2, data3):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(data1, cmap='gray')
    plt.title('Original image')

    plt.subplot(2, 2, 2)
    plt.imshow(data2, cmap='gray', vmin=0, vmax=255)
    plt.title('J1 image')

    plt.subplot(2, 2, 4)
    plt.imshow(data3, cmap='gray', vmin=0, vmax=255)
    plt.title('J2 image')

    plt.show()


if __name__ == '__main__':
    main()