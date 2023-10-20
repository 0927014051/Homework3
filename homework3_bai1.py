import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
def read_sec_file(file_name, size):
    try:
        with open(file_name, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8, count=size*size)
    except FileNotFoundError:
        print(f"\nLoi: Khong the mo file {file_name}!")
        return None
    return data

def main():
    xsize = 256
    thresh = 95
    file_name = 'dataset/Mammogrambin.sec'
    x_original = read_sec_file(file_name, xsize)
    if x_original is None:
        return
    x_original = x_original.reshape(xsize, xsize)
    x = 255 * (x_original >= thresh)
    contours, _ = cv2.findContours(x.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.drawContours(np.zeros_like(x, dtype=np.uint8), contours, -1, (255, 255, 255), 2)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(x_original, cmap='gray', norm=NoNorm())
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(x, cmap='gray', norm=NoNorm())
    plt.title('Thresholding Result')

    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='gray', norm=NoNorm())
    plt.title('Contour Image')

    plt.show()

if __name__ == "__main__":
    main()