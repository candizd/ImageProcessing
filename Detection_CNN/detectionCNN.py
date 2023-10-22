# -*- coding: utf-8 -*-

#%%pyramid method

import cv2
import matplotlib.pyplot as plt

def image_pyramid(image, scale = 1.5, minSize = (224,224)):
    
    yield image
    
    while True:
        w = int(image.shape[1] / scale)
        print(w)
        image = cv2.resize(image, dsize=(w,w))
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image
        
        
img = cv2.imread("husky.jpg")
im = image_pyramid(img, 1.5, (10,10))

for i, image in enumerate(im):
    print(i)
    if i == 5:
        plt.imshow(image)
        
#%% sliding window

def sliding_window(image,step, ws) :
    
    for y in range(0, image.shape[0] - ws[1], step): 
        for x in range(0, image.shape[1] - ws[0], step):
            yield(x,y, image[y: y+ws[1] , x: x+ws[0]])

img = cv2.imread("husky.jpg")
im = sliding_window(img, 5, (200,150))

for i, image in enumerate(im):
    if i == 12456:
        print(image[0],image[1])
        plt.imshow(image[2])
        
        
#%% non maximum suppression 

import numpy as np
import cv2

def non_max_suppression(boxes, probs = None, overlapThreshold = 0.3):
    
    if len(boxes ) == 0 :
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
        
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
        
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    if probs is not None:
        idxs = probs
    #sort 
    idxs = np.argsort(idxs)
    
    pick = []
    
    while len(idxs) > 0:
        
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)
       
        # overlap 
        overlap = (w*h)/area[idxs[:last]]
       
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThreshold)[0])))
       
    return boxes[pick].astype("int")

#%% RCNN

from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

# parameters

WIDTH = 600
HEIGHT = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (200,150)
INPUT_SIZE = (224,224)

print("Loading Resnet")
model = ResNet50(weights = "imagenet", include_top = True)

original = cv2.imread("husky.jpg")
original = cv2.resize(original, (WIDTH, HEIGHT))
cv2.imshow("Husky", original)
        
#image pyramid

pyramid = image_pyramid(original, PYR_SCALE, ROI_SIZE)
rois = []
locs = []

for image in pyramid:
    
    scale = WIDTH / float(image.shape[1])
    
    for (x,y,roiOriginal) in sliding_window(image, WIN_STEP, ROI_SIZE):
        
        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)
        
        roi = cv2.resize(roiOriginal, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
    
        rois.append(roi)
        locs.append((x,y,x+w,y+h))
        

rois = np.array(rois, dtype = "float32")


print("sınıflandırma işlemi")
preds = model.predict(rois)


preds = imagenet_utils.decode_predictions(preds, top = 1)

labels = {}
min_conf = 0.9

for (i,p) in enumerate(preds):
    
    (_, label, prob) = p[0]
    
    if prob >= min_conf:
        
        box = locs[i]
        
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L


for label in labels.keys():
    
    clone = original.copy()
    
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY),(endX, endY), (0,255,0),2)
    
    cv2.imshow("ilk",clone)
    
    clone = original.copy()
    
    # non-maxima
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    
    boxes = non_max_suppression(boxes, proba)
    
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY),(endX, endY), (0,255,0),2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX , y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0),2)
        
    cv2.imshow("Maxima", clone)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break
        
        
#%% selective search

import cv2
import random

image = cv2.imread("pyramid.jpg")
image = cv2.resize(image, dsize = (600,600))
cv2.imshow("image", image)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

ss.switchToSelectiveSearchQuality()        
        
print("start")
rectangle = ss.process()
        
output = image.copy()

for (x,y,w,h) in rectangle[:50]:
    color = [random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output, (x,y), (x+w, y+h), color, 2)
    
cv2.imshow("output", output)
    
#%% 

from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

def selective_search(image):
    print("ss")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
	
    ss.switchToSelectiveSearchQuality()
    
    rects = ss.process()
    
    return rects[:1000]


# model
model = ResNet50(weights="imagenet")    
image = cv2.imread("animals.jpg")
image = cv2.resize(image, dsize = (400,400))
(H, W) = image.shape[:2]


# ss
rects = selective_search(image)

proposals = []
boxes = []
for (x, y, w, h) in rects:


    if w / float(W) < 0.1 or h / float(H) < 0.1: continue
    
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))

    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x, y, w, h))


proposals = np.array(proposals)

# predict
print("predict")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

labels = {}
min_conf = 0.8
for (i, p) in enumerate(preds):
    
    
    (_, label, prob) = p[0]
    if prob >= min_conf:
        (x, y, w, h) = boxes[i]
        box = (x, y, x + w, y + h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

clone = image.copy()

for label in labels.keys():
    for (box, prob) in labels[label]:
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
    
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.imshow("After", clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
    
    
#%%Object detection with R-CNN

import cv2
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

image = cv2.imread("mnist.png")
cv2.imshow("Image",image)

# ilklendir ss
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("SS")
rects = ss.process()

proposals = []
boxes = []
output = image.copy()

for (x,y,w,h) in rects[:200]:
    
    color = [random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output, (x,y), (x+w,y+h),color, 2)
    
    roi = image[y:y+h,x:x+w]
    roi = cv2.resize(roi, dsize=(32,32), interpolation = cv2.INTER_LANCZOS4)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    roi = img_to_array(roi)
    
    proposals.append(roi)
    boxes.append((x,y,w+x,h+y))
    
proposals = np.array(proposals, dtype = "float64")    
boxes = np.array(boxes, dtype = "int32")    

print("sınıflandırma")
# pickle_in = open("model_trained_v4.p", "rb")   
# model = pickle.load(pickle_in)
# model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])
model = load_model('my_model.h5')
proba = model.predict(proposals)

number_list = []
idx = []
for i in range(len(proba)):
    
    max_prob = np.max(proba[i,:])
    if max_prob > 0.95:
        idx.append(i)
        number_list.append(np.argmax(proba[i]))
        
for i in range(len(number_list)):
    
    j = idx[i]
    cv2.rectangle(image, (boxes[j,0], boxes[j,1]), (boxes[j,2],boxes[j,3]),[0,0,255],2)
    cv2.putText(image, str(np.argmax(proba[j])),(boxes[j,0] + 5, boxes[j,1] +5 ), cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,0))
    
    cv2.imshow("Image",image)
    
    
#%% Object Detection with Yolo

import cv2
import numpy as np
from yolo_model import YOLO

yolo = YOLO(0.6, 0.5)
file = "data/coco_classes.txt"

with open(file) as f:
    class_name = f.readlines()
    
all_classes = [c.strip() for c in class_name]

f = "dog_cat.jpg"
path = "images/"+f
image = cv2.imread(path)
cv2.imshow("image",image)

pimage = cv2.resize(image, (416,416))
pimage = np.array(pimage, dtype = "float32")
pimage /= 255.0
pimage = np.expand_dims(pimage, axis = 0)

# yolo
boxes, classes, scores = yolo.predict(pimage, image.shape)

for box, score, cl in zip(boxes, scores, classes):
    
    x,y,w,h = box
    
    top = max(0, np.floor(x + 0.5).astype(int))
    left = max(0, np.floor(y + 0.5).astype(int))
    right = max(0, np.floor(x + w + 0.5).astype(int))
    bottom = max(0, np.floor(y + h + 0.5).astype(int))

    cv2.rectangle(image, (top,left), (right, bottom),(255,0,0),2)
    cv2.putText(image, "{} {}".format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)
    
cv2.imshow("yolo",image)    
































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        