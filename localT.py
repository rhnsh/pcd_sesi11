import imageio.v3 as i
import numpy as np
import matplotlib.pyplot as plt

def lt(img,block_size,c):
    iPad=np.pad(img,pad_width=1,mode='constant',constant_values=0)
    thres=np.zeros_like(img)
    for k in range(img.shape[0]):
        for j in range(img.shape[1]):
            l_area=iPad[k:k+block_size,j:j+block_size]
            l_mean=np.mean(l_area)
            thres[k,j]=255 if img[k][j]>(l_mean-c) else 0
    return thres

img = i.imread("C:\\Users\\muham\\OneDrive\\Desktop\\cr.jpg",mode='F')
img2 = i.imread("C:\\Users\\muham\\OneDrive\\Desktop\\cr.jpg")

thres=lt(img,15,10)
mask= (thres==255).astype(np.uint8)
segm=img2*mask[:,:,np.newaxis]

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(img2)

plt.subplot(1,3,2)
plt.imshow(thres,cmap='gray')

plt.subplot(1,3,3)
plt.imshow(segm)

plt.show()