# -*- coding: utf-8 -*-

#%%

import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)

# bir tane frame oku
ret, frame = cap.read()

if ret == False:
    print("Uyarı")
    
# detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])
track_window = (face_x, face_y, w, h) # meanshift algoritması girdisi

# region of interest
roi = frame[face_y:face_y + h, face_x : face_x + w] # roi = face

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180]) # takip için histogram gerekli
cv2.normalize(roi_hist , roi_hist ,0 ,255, cv2.NORM_MINMAX)

# takip icin gerekli durdurma kriterleri
# count = hesaplanacak maksimum oge sayısı
# eps = degisiklik
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # histogramı bir goruntude bulmak için kullnıyoruz
        # piksel karşılaştırma
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180],1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        x,y,w,h = track_window
        
        img2 = cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),5)
        
        cv2.imshow("Takip", img2)
        
        if cv2.waitKey(1) & 0xFF == ord("q"): break
            
cap.release()
cv2.destroyAllWindows()

#%% Exploratory Data Analysis

import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt

pathIn = r"img1"
pathOut = "deneme.mp4"


files = [f for f in os.listdir(pathIn) if isfile(join(pathIn,f))]

# img = cv2.imread(pathIn + "\\" + files[44])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)

fps = 25
size = (1920,1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, True)

for i in files:
    print(i)
    
    filename = pathIn + "\\" + i
    img = cv2.imread(filename)
    out.write(img)
    
out.release()


#%% Multi Object Tracking

import cv2

OPENCV_OBJECT_TRACKERS = {"csrt"      : cv2.TrackerCSRT_create,
		                  "kcf"       : cv2.TrackerKCF_create,
		                  "boosting"  : cv2.legacy.TrackerBoosting_create,
		                  "mil"       : cv2.TrackerMIL_create,
		                  "tld"       : cv2.legacy.TrackerTLD_create,
		                  "medianflow": cv2.legacy.TrackerMedianFlow_create,
		                  "mosse"     : cv2.legacy.TrackerMOSSE_create}

tracker_name = "kcf"

trackers = cv2.legacy.MultiTracker_create()

video_path = "MOT17-04-DPM.mp4"
cap = cv2.VideoCapture(video_path)

fps = 30     
f = 0
while True:
    
    ret, frame = cap.read()
    (H, W) = frame.shape[:2]
    frame = cv2.resize(frame, dsize = (960, 540))
    
    (success , boxes) = trackers.update(frame)
    
    info = [("Tracker", tracker_name),
        	("Success", "Yes" if success else "No")]
    
    string_text = ""
    
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + " "
    
    cv2.putText(frame, string_text, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("t"):
        
        box = cv2.selectROI("Frame", frame, fromCenter=False)
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
        
    elif key == ord("q"):break

    f = f + 1
    
cap.release()
cv2.destroyAllWindows() 
    
#%%
import cv2


OPENCV_OBJECT_TRACKERS = {"csrt"      : cv2.TrackerCSRT_create,
		                  "kcf"       : cv2.TrackerKCF_create,
		                  "boosting"  : cv2.legacy.TrackerBoosting_create,
		                  "mil"       : cv2.TrackerMIL_create,
		                  "tld"       : cv2.legacy.TrackerTLD_create,
		                  "medianflow": cv2.legacy.TrackerMedianFlow_create,
		                  "mosse"     : cv2.legacy.TrackerMOSSE_create}

tracker_name = "medianflow"

video_path = "MOT17-04-DPM.mp4"
cap = cv2.VideoCapture(video_path)

fps = 30
f = 0

# Create an empty dictionary to store trackers
trackers = {}

while True:

    ret, frame = cap.read()
    (H, W) = frame.shape[:2]
    frame = cv2.resize(frame, dsize=(960, 540))

    for object_id, tracker in trackers.items():
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    info = [("Tracker", tracker_name),
            ("Number of Objects", len(trackers))]

    string_text = ""

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + " "

    cv2.putText(frame, string_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("t"):
        box = cv2.selectROI("Frame", frame, fromCenter=False)
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        tracker.init(frame, box)
        trackers[f] = tracker
        f += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
    

    
    
    
    
    
    
    
    
    
    





















