import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import *
from typing import List, Dict, Tuple
from numpy._typing import *

def exp2cosine(x: ArrayLike) -> ArrayLike:
    '''convert e^jx -> cos(x) + j*sin(x) '''
    return np.cos(x) + 1j*np.sin(x) 

# def fft2D(img):
#     '''Hàm này chưa được tối ưu RAM, chạy sẽ bị tràn RAM'''
#     img = img.T
#     M, N= img.shape[:2]
#     u = x = np.arange(M)
#     v = y = np.arange(N)  
#     ux = np.expand_dims(
#                 u[:, None]*x[None,:],
#                 axis = -1    
#             ) 
#     vy = (v[:, None]*y[None,:]).reshape(1,-1)
    
#     return np.round(
#         np.sum(
#             img*exp2cosine(
#                 -2*pi*(
#                     (ux * 1.0/M + vy*1.0/N).reshape((M,M,N,N)).transpose(0,2,1,3)
#                 )
#             ), 
#             axis = (-1,-2)
#         )*1.0/(M*N), 4
#     ).T

def fft2D(img: ArrayLike) -> ArrayLike:
    ft = np.zeros_like(img, dtype = np.complex64)
    T_img = img.T
    M, N= T_img.shape[:2]
    x = u = np.arange(M)
    y = np.arange(N)
    for v in range(img.shape[0]):
        s = T_img*exp2cosine(
            -2*pi*(
                np.expand_dims(
                    u[:, None]*x[None,:],
                    axis = -1    
                ) * 1.0/M + v*y[None,:]*1.0/N
            )
        )
        ft[v,:] = np.sum(
            s, axis = (-1,-2)
        )
            
    return ft

def ifft2D(img: ArrayLike) -> ArrayLike:
    ft = np.zeros_like(img, dtype = np.complex64)
    T_img = img.T
    M, N= T_img.shape[:2]
    x = u = np.arange(M)
    y = np.arange(N)
    for v in range(img.shape[0]):
        s = T_img*exp2cosine(
            2*pi*(
                np.expand_dims(
                    u[:, None]*x[None,:],
                    axis = -1    
                ) * 1.0/M + v*y[None,:]*1.0/N
            )
        )
        ft[v,:] = np.sum(
            s, axis = (-1, -2)
        )*1.0/(M*N)
            
    return ft


img = cv.imread('test.jpg', 0)

### Sử dụng hàm đã được code ở trên ###
fft_code = fft2D(img)
fshift = np.fft.fftshift(fft_code)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.figure(figsize = (15,10))
plt.subplot(241)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])

plt.subplot(242)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])

M, N = img.shape
center = M//2, N//2
R = 25
x, y = np.ogrid[:M, :N]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= R**2
fshift[mask_area] = 0
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(243)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('High pass filters in frequency domain')
plt.xticks([]), plt.yticks([])

ifshift = np.fft.ifftshift(fshift)
ifft_code = ifft2D(ifshift)
ifft_img = np.abs(ifft_code)
plt.subplot(244)
plt.imshow(ifft_img, cmap = 'gray')
plt.title('highpass filtered image')
plt.xticks([]), plt.yticks([])

plt.show()

### Sử dụng thư viện numpy ###

fft_np = np.fft.fft2(img)
fshift = np.fft.fftshift(fft_np)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.figure(figsize = (15,10))

plt.subplot(245)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])

plt.subplot(245)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])

M, N = img.shape
center = M//2, N//2
R = 25
x, y = np.ogrid[:M, :N]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= R**2
fshift[mask_area] = 0
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(245)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('High pass filters in frequency domain')
plt.xticks([]), plt.yticks([])

ifft_shift = np.fft.ifftshift(fshift)
ifft = np.fft.ifft2(ifft_shift)
plt.subplot(245)
plt.imshow(np.abs(ifft), cmap = 'gray')
plt.title('highpass filtered image')
plt.xticks([]), plt.yticks([])
plt.show()