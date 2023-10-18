# -*- coding: utf-8 -*-
#%% data collecting
import keyboard 
import uuid
import time
from PIL import Image
from mss import mss

def exit():
    
    global is_exit
    is_exit = True
    
def record_screen(recordID, key) :
    global i
    i+= 1
    print("{}: {}".format(key,i))
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key,recordID,i))
    
    
mon = {"top": 400, "left": 750, "width" : 250, "height" : 100}

sct = mss()

i = 0

is_exit = False

keyboard.add_hotkey("esc", exit)
recordID = uuid.uuid4()

while True:
    if is_exit: break
    
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(recordID, "up")
            time.sleep(0.5)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(recordID, "down")
            time.sleep(0.5)
        elif keyboard.is_pressed("right"):
            record_screen(recordID, "right")
            time.sleep(0.5)

    except RuntimeError: continue

#%% Training the model

import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

imgs = glob.glob("./img/*.png")

width = 125
height = 50

X = []
Y = []

for img in imgs:
    
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im / 255
    X.append(im)
    Y.append(label)
    
X = np.array(X)
X = X.reshape(-1, width, height, 1)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Y = onehot_labels(Y)
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)    

# cnn model
model = Sequential()   
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))

# if os.path.exists("./trex_weight.h5"):
#     model.load_weights("trex_weight.h5")
#     print("Weights yuklendi")    

model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

model.fit(train_X, train_y, epochs = 35, batch_size = 64)

score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %",score_train[1]*100)    
    
score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %",score_test[1]*100)      
    
open("model_new.json","w").write(model.to_json())
model.save_weights("trex_weight_new.h5")   

#not a good model overall although I have really low sample size. 

#%% Auto play 

#Didn't work well because I couldn't sync the delay and sleep. 

from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

def exit():
    
    global is_exit
    is_exit = True

is_exit = False

mon = {"top":400, "left":750, "width":250, "height":100}
sct = mss()

width = 125
height = 50

# model yükle
model = model_from_json(open("model_new.json","r").read())
model.load_weights("trex_weight_new.h5")

# down = 0, right = 1, up = 2
labels = ["Down", "Right", "Up"]

framerate_time = time.time()
counter = 0
i = 0
delay = 0.4
key_down_pressed = False

while True:
    if is_exit: break

    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255
    
    X =np.array([im2])
    X = X.reshape(-1, width, height, 1)
    r = model.predict(X)
    
    result = np.argmax(r)
    
    
    if result == 0: # down = 0
        
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
        
    elif result == 2:    # up = 2
        
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        
        if i < 1500:
            time.sleep(0.3)
        elif 1500 < i and i < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)
            
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
    
    counter += 1
    
    if (time.time() - framerate_time) > 1:
        
        counter = 0
        framerate_time = time.time()
        if i <= 1500:
            delay -= 0.003
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0
            
        print("---------------------")
        print("Down: {} \nRight:{} \nUp: {} \n".format(r[0][0],r[0][1],r[0][2]))
        i += 1
        









































































