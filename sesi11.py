import imageio.v3 as i
import matplotlib.pyplot as plt
import numpy as np

img = i.imread("C:\\Users\\muham\\OneDrive\\Desktop\\cr.jpg", mode='F') 

sx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sy = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

ipad = np.pad(img, pad_width=1, mode='constant', constant_values=0) 
gx = np.zeros_like(img)
gy = np.zeros_like(img)

for y in range(1, ipad.shape[0] - 1):
    for x in range(1, ipad.shape[1] - 1):
        area = ipad[y-1:y+2, x-1:x+2]
        gx[y-1, x-1] = np.sum(area * sx)
        gy[y-1, x-1] = np.sum(area * sy)


G = np.sqrt(gx**2 + gy**2)
G = (G / G.max()) * 255  
G = np.clip(G, 0, 255).astype(np.uint8) 

sobel_path = "C:\\Users\\muham\\OneDrive\\Desktop\\sobel.jpg"
i.imwrite(sobel_path, G)

def lt(img, block_size, c):
    iPad = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    thres = np.zeros_like(img)
    for k in range(img.shape[0]):
        for j in range(img.shape[1]):
            l_area = iPad[k:k+block_size, j:j+block_size]
            l_mean = np.mean(l_area)
            thres[k, j] = 255 if img[k, j] > (l_mean - c) else 0
    return thres


thres = lt(G, 15, 10)
mask = (thres == 255).astype(np.uint8)


img2 = i.imread("C:\\Users\\muham\\OneDrive\\Desktop\\cr.jpg")
segm = img2 * mask[:, :, np.newaxis]  

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img2)

plt.subplot(2, 3, 2)
plt.title("Sobel Gx")
plt.imshow(gx, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("Sobel Gy")
plt.imshow(gy, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("Sobel Magnitude")
plt.imshow(G, cmap='gray')

plt.subplot(2, 3, 5)
plt.title("Local Thresholding")
plt.imshow(thres, cmap='gray')

plt.subplot(2, 3, 6)
plt.title("Segmented Image")
plt.imshow(segm)

plt.tight_layout()
plt.show()
