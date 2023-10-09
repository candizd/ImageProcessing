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
            M = cv2.moment(c)
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
    
    













































