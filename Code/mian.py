# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:30:17 2017

@author: Jadon
"""

import numpy as np
import glob
#from skimage import io
#import tensorflow as tf
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing.image import  img_to_array
from PIL import Image
from skimage import io
import dlib
import coin_detector
import os
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')

def get_val(st):
    v=0.0
    if st == "10C":
        v=0.10
    elif st == "1R":
        v=1.0
    elif st == "20C":
        v=0.20
    elif st == "2R":
        v=2.0
    elif st == "50C":
        v=0.50
    elif st == "5R":
        v=5.0
    return v
def make_prediction(img, model):
    m,n=100,100
    im = Image.open(img)
    im=im.convert(mode="RGB")
    imrs = im.resize((m,n))
    imrs=img_to_array(imrs)/255
    imrs=imrs.transpose(2,0,1)
    imrs=imrs.reshape(3,m,n)
    
    x=[]
    x.append(imrs)
    x=np.array(x)
    predictions = model.predict(x)
    
    label=np.argmax(predictions)
    return label    

def box_colour(label):
    #using RGB - skimage io
    col=(255,255,255)
    
    #5R
    red =(255,0,0)
    #2R
    yellow = (255,255,0)
    #1R
    blue=(0,0,255)
    
    #50C
    green = (0,255,0)
    #20C
    purple = (255,0,255)
    #10C
    black= (0,0,0)
    
    if label == "10C":
        col = black
    elif label == "1R":
        col = blue
    elif label == "20C":
        col = purple
    elif label == "2R":
        col = yellow
    elif label == "50C":
        col = green
    elif label == "5R":
        col = red
    return col

def update_coins(label, nb_c_ten, nb_c_twen, nb_c_fif, nb_r_one, nb_r_two, nb_r_fiv):
    if label == "10C":
        nb_c_ten +=1
    elif label == "1R":
        nb_r_one += 1
    elif label == "20C":
        nb_c_twen +=1
    elif label == "2R":
        nb_r_two += 1
    elif label == "50C":
        nb_c_fif += 1
    elif label == "5R":
        nb_r_fiv += 1
    return nb_c_ten, nb_c_twen, nb_c_fif, nb_r_one, nb_r_two, nb_r_fiv

def see_coins(img_path, boxes, cols):
    img = io.imread(img_path)
    #draw each box
    win = dlib.image_window()
    win.set_image(img)
    for i in np.arange(len(boxes)):
        win.add_overlay(boxes[i], color=dlib.rgb_pixel(cols[i][0], cols[i][1], cols[i][2]))
    #plt.figure()
    #plt.imshow(img)
    
    win.wait_until_closed() 
    #time.sleep(10)
    
    
def write_val_to_file(val, dest):
    #put value in the filename 
    v=str(val)+".txt"
    filename = os.path.join(dest, v)
    with open(filename, "w") as f: 
        f.write(str(val)) 
    
    
def process_imgs(folder_pth, detector_pth, c_model_pth, c_weights_pth):
    
    #Get path to all image files as list
    path_to_imgs = folder_pth + "*.jpg"
    files = glob.glob(path_to_imgs)
    
    #Coin Classes
    classes=["10C","1R","20C","2R","50C","5R"]
    
    #Load CNN Model
    json_file = open(c_model_pth, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    # load weights into new model
    loaded_model.load_weights(c_weights_pth)
    print("Loaded model from disk")
    print()
     
 
    for i in np.arange(len(files)):
        #List to hold respective colors of coin detections
        box_cols=[]
        
        #Split files path and name
        s = files[i].split("\\")
        #get files full name.jpg
        name = s[-1]
        #Get files name without .jpg
        n = name.split(".jpg")
        nj=n[0]
        
        #Set save destination to be same path as image name
        dest = os.path.join(s[0], str(nj))
        
        #first check if dir allready exsists
        
        #Run coin detector on image and save detections to folder with image name
        detections, boxs = coin_detector.detect_crop(files[i], dest, detector_pth)
        
        #then for each detected coin, classify value
        val =0.0
        
        nb_c_ten=0
        nb_c_twen=0
        nb_c_fif =0
        nb_r_one=0
        nb_r_two=0
        nb_r_fiv=0
        
        for j in np.arange(len(detections)):
            
            #Use the CNN model to make a prediction
            label = make_prediction(detections[j], loaded_model)
            #Get the actual float value of the prediction
            l_val= get_val(classes[label])
            #Add to total val of image
            val +=l_val
            
            #hold label of predicted class
            l = classes[label]
            
            #update number of coins per class seen
            nb_c_ten, nb_c_twen, nb_c_fif, nb_r_one, nb_r_two, nb_r_fiv = update_coins(l,nb_c_ten, nb_c_twen, nb_c_fif, nb_r_one, nb_r_two, nb_r_fiv)
            
            #print("label : "+str(classes[label]))
            #print("val : "+str(l_val))
            
            
            #Use label to assign box colour
            col = box_colour(l)
            box_cols.append(col)
            
        print("Total Value "+str(val))
        print("Num R5 : "+str(nb_r_fiv))
        print("Num R2 : "+str(nb_r_two))
        print("Num R1 : "+str(nb_r_one))
        print("Num 50C : "+str(nb_c_fif))
        print("Num 20C : "+str(nb_c_twen))
        print("Num 10C : "+str(nb_c_ten))
        
        #Write the final value to the folder
        write_val_to_file(val, dest)
        
        
        see_coins(files[i], boxs, box_cols)
    
def confused():
    pass

if __name__ == "__main__":
    

    #folder containing imgs to detect and classify
    fold = "../data/Val/"
    
    #detector to use
    detector = "../data/TrainHOG/real_coindetective.svm"
    
    #classifier to use
    cnn_model = "../data/cvmodel/model_4l.json"
    cnn_weights = "../data/cvmodel/model_4l.h5"
    
    #Process images in folder
    
    process_imgs(fold, detector, cnn_model, cnn_weights)
    
    
    
    
    
    
    
    
    