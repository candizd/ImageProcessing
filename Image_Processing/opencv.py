# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:10:45 2023

@author: hasan
"""
#%% Resmi içe aktarma

import cv2 

image = cv2.imread("fieldGround.png",0)

#görselleştir

cv2.imshow("Ilk Resim", image)

k = cv2.waitKey(0) &0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("ground_Gray.png",image)
    cv2.destroyAllWindows()

#%% Video İçe Aktarma

import time

videoName= "MOT17-04-DPM.mp4"

cap = cv2.VideoCapture(videoName)

print("Genişlik: ", cap.get(3))
print("Yükseklik: ", cap.get(4))

if cap.isOpened() == False:
    print("Error")
    

while True:
    
    ret, frame = cap.read()
    
    if ret == True:
        time.sleep(0.01) # attention: if we don't use this method the images will go by too fast 
        
        cv2.imshow("Video", frame)
    else:
        break
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()


#%% Kamera Açma ve Video Kaydı

#kameram olmadığı için denemiyorum

#%% Yeniden boyutlandır ve Kırp

# Resize
img = cv2.imread("lenna.png")
print("Resim boyutu", img.shape)
cv2.imshow("Orijinal: ", img)

imgResized = cv2.resize(img, (800,800))
print("Resized Img Shape: ", imgResized.shape)
cv2.imshow("Resized: ", imgResized)

# Crop

imgCropped = img[:200,0:300]
cv2.imshow("Cropped Img", imgCropped)

#%% Shape Text

import numpy as np

#Resim oluştur

img = np.zeros((512,512,3), np.uint8) # siyah bir resim

print(img.shape)

cv2.imshow("Siyah", img)

# çizgi
# (resim, başlangıç noktası, bitiş noktası, renk, kalınlık )
cv2.line(img, (0,0), (512,512), (0,255,0), 3) 
#cv2.imshow("Cizgi", img)

# Rectangle
# (resim,başlangıç,bitiş,renk)
cv2.rectangle(img, (0,0), (256,256), (255,0,0),cv2.FILLED)
#cv2.imshow("Rectangle", img)

#çember
#(resim, merkez, yarıçap,renk)
cv2.circle(img, (400,300), 45, (0,0,255),cv2.FILLED)
#cv2.imshow("Circle", img)

#metin
#(resim,text,başlangıç,font,kalınlık,renk)
cv2.putText(img,"Resim", (200,450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1, (255,255,255))
cv2.imshow("Combined Image",img)

#%% Joining Images

img = cv2.imread("lenna.png")

#yatay
hor = np.hstack((img,img))
cv2.imshow("Horizontal", hor)

#dikey
ver = np.vstack((img,img))
cv2.imshow("Vertical", ver)

#%% Warping Perspective
import cv2 
import numpy as np

img = cv2.imread("kart.png")
print(img.shape)
width = 500
height = 544

pts1 = np.float32([[203,1],[1,472],[540,150], [338,617]])
pts2 = np.float32([[0,0],[0,height],[width,0],[width, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

imgOutput = cv2.warpPerspective(img, matrix, (width,height))
cv2.imshow("Final Image", imgOutput)

#%% blending

import matplotlib.pyplot as plt

img1 = cv2.imread("img1.jpg")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.imread("img2.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1,(600,600))
img2 = cv2.resize(img2,(600,600))

#karıştırılmış resim = alpha * image1 + beta * image2
blended = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma = 0)

plt.figure()
plt.imshow(blended)



#%% image thresholding

img = cv2.imread("img1.jpg",0)


plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

#thresholding
_, thresh_img = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img, cmap="gray")
plt.axis("off")
plt.show()

#adaptive thresholding

thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13 , 8)
plt.figure()
plt.imshow(thresh_img2, cmap="gray")
plt.axis("off")
plt.show()

#%% Blurring

import warnings
warnings.filterwarnings("ignore")

def gaussianNoise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma, (row, col, ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def saltPepperNoise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    noisy = np.copy(image)

    # Salt noise (white pixels)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:-1]]
    noisy[salt_coords[0], salt_coords[1], :] = 255  # 255 represents white

    # Pepper noise (black pixels)
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:-1]]
    noisy[pepper_coords[0], pepper_coords[1], :] = 0  # 0 represents black

    return noisy

#blurring (detayı, azaltır, gürültüyü engeller)

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("orijinal"), plt.show()

#1st method (Mean)

dst2 = cv2.blur(img, ksize = (3,3))
plt.figure(), plt.imshow(dst2), plt.axis("off"), plt.title("Mean")

#2st method (Gauss)

gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb), plt.axis("off"), plt.title("Gaussian")

#3rd method (Median)

mb = cv2.medianBlur(img, ksize = 3)
plt.figure(), plt.imshow(mb), plt.axis("off"), plt.title("Median")


#normalize et
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("original"), plt.show()

noisyImage = gaussianNoise(img)
plt.figure(), plt.imshow(noisyImage), plt.axis("off"), plt.title("Noisy Image"), plt.show()

#blur the noisy image to reduce noise

gbNoisy = cv2.GaussianBlur(noisyImage, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(gbNoisy), plt.axis("off"), plt.title("Gaussian")

#salt pepper image
spicyImage = saltPepperNoise(img)
plt.figure(), plt.imshow(spicyImage), plt.axis("off"), plt.title("Seasoned"), plt.show()


mb2 = cv2.medianBlur(spicyImage, ksize = 3)
plt.figure(), plt.imshow(mb2), plt.axis("off"), plt.title("Median")


#%% morphological operations

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("datai_team.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("Original")

#erosion 

kernel = np.ones((7,7), dtype = np.uint8)
result = cv2.erode(img,kernel,iterations = 1)
plt.figure(), plt.imshow(result, cmap = "gray"), plt.axis("off"), plt.title("Erosion")

#dilation 

result2 = cv2.dilate(img, kernel, iterations = 1 )
plt.figure(), plt.imshow(result2, cmap = "gray"), plt.axis("off"), plt.title("Dilation")

#while noise
whiteNoise = np.random.randint(0,2, size = img.shape)
whiteNoise = whiteNoise * 255 
noisyImage = whiteNoise + img
plt.figure(), plt.imshow(noisyImage, cmap = "gray"), plt.axis("off"), plt.title("White Noise")

#opening

opening = cv2.morphologyEx(noisyImage.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Opening")

#black noise

blackNoise = np.random.randint(0,2, size = img.shape)
blackNoise = blackNoise * -255 
noisyImage = blackNoise + img
noisyImage[noisyImage <= -245] = 0
plt.figure(), plt.imshow(noisyImage, cmap = "gray"), plt.axis("off"), plt.title("Black Noise")

#closing

closing = cv2.morphologyEx(noisyImage.astype(np.float32), cv2.MORPH_CLOSE, kernel)
plt.figure(), plt.imshow(closing, cmap = "gray"), plt.axis("off"), plt.title("Closing")

# gradient / edge detection

gradient = cv2.morphologyEx(img.astype(np.float32), cv2.MORPH_GRADIENT, kernel)
plt.figure(), plt.imshow(gradient, cmap = "gray"), plt.axis("off"), plt.title("Gradient")


#%% Gradients

img = cv2.imread("sudoku.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("Original")

# gradient x

sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.imshow(sobelx, cmap = "gray"), plt.axis("off"), plt.title("Sobelx")

# gradient y 

sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobely, cmap = "gray"), plt.axis("off"), plt.title("Sobely")

#laplacian 

laplacian = cv2.Laplacian(img, ddepth= cv2.CV_16S)
plt.figure(), plt.imshow(laplacian, cmap = "gray"), plt.axis("off"), plt.title("Laplacian")



#%% histograms

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img)

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(img_hist)

color = ("b","g","r")
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels = [i], mask = None, histSize = [256], ranges = [0,256])
    plt.plot(hist,color = c)


#

rawGoldenGate = cv2.imread("goldenGate.jpg")
goldenGate = cv2.cvtColor(rawGoldenGate,cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(goldenGate)

mask = np.zeros(rawGoldenGate.shape[:2], np.uint8)
plt.figure(), plt.imshow(mask, cmap = "gray")


mask[1500:2000, 1000:2000] = 255
plt.figure(), plt.imshow(mask, cmap = "gray")

masked_img_vis = cv2.bitwise_and(goldenGate,goldenGate, mask = mask)
plt.figure(), plt.imshow(masked_img_vis)

masked_img_hist = cv2.calcHist([goldenGate], channels = [0], mask = mask, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(masked_img_hist)

# histogram eşitleme | kontrastı arttırma

img = cv2.imread("hist_equ.jpg",0)
plt.figure(), plt.imshow(img,cmap="gray")

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(img_hist)

eq_img = cv2.equalizeHist(img)
plt.figure(), plt.imshow(eq_img,cmap="gray")

eq_img_hist = cv2.calcHist([eq_img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(eq_img_hist)






























