# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:45:29 2023

@author: hasan
"""
#%% Edge Detection

import cv2 
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("london.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

#Without doing anything
edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
plt.figure(), plt.imshow(edges, cmap="gray"), plt.axis("off")

#With thresholds
medImg = np.median(img)
print(medImg)

low = int(max(0, (1 - 0.33) * medImg))
high = int(min(255, (1 + 0.33) * medImg))

print(low, high)

edges = cv2.Canny(image = img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap="gray"), plt.axis("off")

#With blurring + thresolds

blurredImg = cv2.blur(img, ksize= (5,5))
plt.figure(), plt.imshow(blurredImg, cmap="gray"), plt.axis("off")

medBlurredImg = np.median(blurredImg)

low = int(max(0, (1 - 0.33) * medBlurredImg))
high = int(min(255, (1 + 0.33) * medBlurredImg))

edges2 = cv2.Canny(image = blurredImg, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges2, cmap="gray"), plt.axis("off")

#%% Corner Detection

import cv2 
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)

plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

# harris corner detection
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
dst = cv2.dilate(dst,None)
img[dst > 0.2 * dst.max()] = 255
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

#shi tomasi detection

img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)

corners = cv2.goodFeaturesToTrack(img, 120, 0.01,10)
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, (125,125,125), cv2.FILLED)

plt.imshow(img), plt.axis("off")

#%% contour detection

img = cv2.imread("contour.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)

for i in range(len(contours)):
    
    #external 
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour, contours, i, 255, -1)
    else:
        cv2.drawContours(internal_contour, contours, i, 255, -1)


plt.figure(), plt.imshow(external_contour, cmap="gray"), plt.axis("off")
plt.figure(), plt.imshow(internal_contour, cmap="gray"), plt.axis("off")


#Object detection with color

import cv2 
import numpy as np
from collections import deque

buffer_size = 16
pts = deque(maxlen = buffer_size)

#HSV blue

blueLower = (84,98,0)
blueUpper = (179,255,255)

#capture

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    
    success, imgOriginal = cap.read()
    
    if success:
        
        #blur
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0)
        
        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)
        
        #mask for blue
        mask = cv2.inRange(hsv,blueLower,blueUpper)
        mask = cv2.erode(mask,None, iterations = 2)
        mask = cv2.dilate(mask,None, iterations = 2)
        cv2.imshow("Mask + erosion and dilate", mask)
        
        #contours
        contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        
        if len(contours) > 0:
            
            #get the max contour
            c = max(contours, key = cv2.contourArea)
            #rectangle
            rect = cv2.minAreaRect(c)
            ((x,y), (width,height), rotation) = rect 
            
            #box
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            #moment
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"]))
            
            #draw          
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255), 2)
            cv2.circle(imgOriginal, center, 5, (255,0,255), -1)
        
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i - 1] is None or pts[i] is None: continue
            cv2.line(imgOriginal, pts[i-1], pts[i], (0,255,0), 3)
        
        
        cv2.imshow("Original", imgOriginal)
        if cv2.waitkey(1) & 0XFF == ord("q") : break
    
cap.release()
cv2.detroyAllWindows()
    
#%% Template Matching

import cv2
import matplotlib.pyplot as plt


img = cv2.imread("cat.jpg",0)
print(img.shape)

template = cv2.imread("cat_face.jpg", 0)
print(template.shape)
h,w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for met in methods:
    
    method = eval(met)
    res = cv2.matchTemplate(img,template,method)
    print(res.shape)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res,cmap="gray")
    plt.title("Matched Image"), plt.axis("off")
    plt.subplot(122), plt.imshow(img,cmap="gray")
    plt.title("Detected Result"), plt.axis("off")
    plt.suptitle(met)
    plt.show()

#%% Feature Matching

chocos = cv2.imread("chocolates.jpg",0)
plt.figure(), plt.imshow(chocos, cmap="gray"), plt.axis("off")

whiteChoco = cv2.imread("nestle.jpg", 0)
plt.figure(), plt.imshow(whiteChoco, cmap="gray"), plt.axis("off")

#Brute Force (doesn't work well and its slow)
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(whiteChoco, None)
kp2, des2 = orb.detectAndCompute(chocos, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x: x.distance)

plt.figure()
img_match = cv2.drawMatches(whiteChoco, kp1, chocos, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off")


#sift

sift = cv2.SIFT_create()

#bf 

bf = cv2.BFMatcher()

# key point detector with sift

kp1, des1 = sift.detectAndCompute(whiteChoco, None)
kp2, des2 = sift.detectAndCompute(chocos, None)

matches = bf.knnMatch(des1,des2, k = 2)

betterMatches = []

for match1, match2 in matches:
    
    if match1.distance < 0.75*match2.distance:
        betterMatches.append([match1])

plt.figure()
sift_matches = cv2.drawMatchesKnn(whiteChoco, kp1, chocos, kp2, betterMatches, None, flags = 2)
plt.imshow(sift_matches), plt.axis("off")

#%% Watershed algorithm

import cv2 
import matplotlib.pyplot as plt
import numpy as np


coins = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coins), plt.axis("off")

#blurring
blurredCoin = cv2.medianBlur(coins,13)
plt.figure(), plt.imshow(blurredCoin), plt.axis("off")

#grayscale

grayCoin = cv2.cvtColor(blurredCoin, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(grayCoin, cmap = "gray"), plt.axis("off")

#binary threshold 

ret, coin_thresh = cv2.threshold(grayCoin, 75,255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

# contour 

contours, hier = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):

    if hier[0][i][3] == -1:
        cv2.drawContours(coins,contours,i,(0,255,0),10)


plt.figure(), plt.imshow(coins, cmap = "gray"), plt.axis("off")

#the upper method didn't work well thought all the coins as one entity

coins = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coins), plt.axis("off")

# lpf: blurring
coin_blur = cv2.medianBlur(coins, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")

# grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")

# binary threshold
ret, coin_thresh = cv2.threshold(coin_gray, 65, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

# opening

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
plt.figure(), plt.imshow(opening, cmap="gray"), plt.axis("off")

#distances between images 

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure(), plt.imshow(dist_transform, cmap="gray"), plt.axis("off")

#minimize image

ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform), 255,0)
plt.figure(), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off")

#enlarge the image for the background

sure_background = cv2.dilate(opening, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background, sure_foreground )
plt.figure(), plt.imshow(unknown, cmap="gray"), plt.axis("off")

#connection

ret, marker = cv2.connectedComponents(sure_foreground)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

marker = marker + 1
marker[unknown == 255] = 0
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

#watershed
marker = cv2.watershed(coins,marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

# contour 

contours, hier = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):

    if hier[0][i][3] == -1:
        cv2.drawContours(coins,contours,i,(255,0,0),10)


plt.figure(), plt.imshow(coins), plt.axis("off")


#%% face recognition

import cv2
import matplotlib.pyplot as plt
import numpy as np

einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

#classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

#barce

barce = cv2.imread("barcelona.jpg", 0)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")

face_rect = face_cascade.detectMultiScale(barce, minNeighbors= 7)
for (x,y,w,h) in face_rect:
    cv2.rectangle(barce, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
            
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),10)
        cv2.imshow("face detect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()


#%% cat face recognition

import cv2
import os 

files = os.listdir()
print(files)

img_list = []
min_size = (70,70)  
max_size = (200, 200)

for f in files :
    if f.startswith("cat_img"): 
        img_list.append(f)
print(img_list)

for j in img_list:
    print(j)
    image = cv2.imread(j)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.030, minNeighbors = 2, minSize = min_size, maxSize = max_size)
    
    for(i,(x,y,w,h)) in enumerate(rects): 
        
        cv2.rectangle(image, (x,y),(x+w, y+h),(0,255,255),2)
        cv2.putText(image, "Kedi {}".format(i+1), (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.55, (0,255,255), 2)
    
    
    cv2.imshow(j, image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue

cv2.destroyAllWindows()

#%% custom cascade

import cv2
import os

path = "images"

imgWidth = 180
imgHeight = 120

# video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180)

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    os.makedirs(path+str(countFolder))

saveDataFunc()

count = 0
countSave = 0

while True:
    
    success, img = cap.read()
    
    if success:
        
        img = cv2.resize(img, (imgWidth, imgHeight))
        
        if count % 5 == 0:
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png",img)
            countSave += 1
            print(countSave)
        count += 1
        
        cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()



#%% Pedestrian detection

files = os.listdir()
img_list = []

for f in files :
    if f.startswith("img"):
        img_list.append(f)
        
print(img_list)

# hog 
hog = cv2.HOGDescriptor()
#svm 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_list:
    image = cv2.imread(imagePath)
    
    (rects, weights) = hog.detectMultiScale(image, padding = (10,10), scale = 1.05)
    
    for (x,y,w,h) in rects:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
         
    cv2.imshow("Pedestrian: ", image)
      
    if cv2.waitKey(0) & 0xFF == ord("q"): continue


cv2.destroyAllWindows()

#%% HA

# opencv kütüphanesini içe aktaralım
# ...
import cv2
# numpy kütüphanesini içe aktaralım
# ...
import numpy as np
import matplotlib.pyplot as plt
# resmi siyah beyaz olarak içe aktaralım resmi çizdirelim
# ...
image = cv2.imread("odev2.jpg", 0)
cv2.imshow("Original", image)
# resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim edge detection
# ...

medImg = np.median(image)
print(medImg)

low = int(max(0, (1 - 0.33) * medImg))
high = int(min(255, (1 + 0.33) * medImg))
edges = cv2.Canny(image = image, threshold1 = low, threshold2 = high)
cv2.imshow("Edges", edges)

# yüz tespiti için gerekli haar cascade'i içe aktaralım
# ...
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# yüz tespiti yapıp sonuçları görselleştirelim
# ...
min_size = (20,20)  
max_size = (120,120)
rects = detector.detectMultiScale(image, minNeighbors= 4, minSize = min_size, maxSize = max_size)
for (x,y,w,h) in rects:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,255), 5)
     
cv2.imshow("Rectangles ", image)
plt.figure(), plt.imshow(image, cmap="gray"), plt.axis("off")
# HOG ilklendirelim insan tespiti algoritmamızı çağıralım ve svm'i set edelim
# ...
image2 = image.copy()
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# resme insan tespiti algoritmamızı uygulayalım ve görselleştirelim
# ...

(rects, weights) = hog.detectMultiScale(image, padding = (10,10), scale = 1.10)
 
for (x,y,w,h) in rects:
     cv2.rectangle(image2, (x,y), (x+w, y+h), (0,0,255), 2)
      
cv2.imshow("HOG: ", image2)




































