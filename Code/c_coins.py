# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:56:14 2017

@author: Jadon
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, errno


def hough_to_circles(image_name):
    training_dir = "../data/Masked/Labels/imgs/"
    masked_dir = "../data/Masked/Labels/"
    
    
    full_pic = training_dir+"t"+img_name+".jpg"    
    
    img = cv2.imread(full_pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # detect circles
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=5, maxRadius=50)
    circles = np.uint16(np.around(circles))
    
    # draw mask
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only 
    for i in circles[0, :]:
        cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
    
    # get first masked value (foreground)
    fg = cv2.bitwise_or(img, img, mask=mask)
    
    # get second masked value (background) mask must be inverted
    mask = cv2.bitwise_not(mask)
    background = np.full(img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)
    
    # combine foreground+background
    final = cv2.bitwise_or(fg, bk)
    plt.imshow(final)
    return final

#helper function to add seperate dirs for each type of coin in each 
#image folder so to make things easier

def help_out_folders():
    dir_tar = "../data/Generated/coins/"
    dir_src = "../data/Generated/imgs/"
    coin_dirs = ["10C/","20C/","50C/","1R/", "2R/", "5R/", "Huh/"]
    for i in np.arange(105):
        #create each sub dir
        for j in np.arange(len(coin_dirs)):
            try:
                f = directory+"img"+str(i)+"/"+str(coin_dirs[j])
                os.makedirs(f)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            
def help_out_folders():
    directory = "../data/Generated/imgs/"
    coin_dirs = ["10C/","20C/","50C/","1R/", "2R/", "5R/", "Huh/"]
    for i in np.arange(105):
        #create each sub dir
        for j in np.arange(len(coin_dirs)):
            try:
                f = directory+"img"+str(i)+"/"+str(coin_dirs[j])
                os.makedirs(f)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            


def make_dir(img_name):
    directory = "../data/Generated/"+str(img_name)+"/"
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return directory

def save_using_bb(i, og_img, masked_img):
    d = make_dir(i)
    img, contours, heirh = cv2.findContours(masked_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #copy og img to folder for reference
    h = d+"og.jpg"
    cv2.imwrite(h, og_img)
    idx =0 
    for i in np.arange(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        roi=og_img[y:y+h,x:x+w]
        place = d+str(idx)
        cv2.imwrite(place + '.jpg', roi)
        idx += 1
        #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
    #plt.imshow(im)
    #cv2.waitKey(0)    
        

#Takes in image name and uses its mask segment and save the images as seperate coins
def segment_using_masks(iden, img_n, msk_n):

    
    
    img = cv2.imread(img_n)
    img_msk = cv2.imread(msk_n,0)
   
    #Mask out
    res = cv2.bitwise_and(img,img,mask = img_msk)
    #convert to greyscale
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    #Threshold all non black pixels
    ret, th = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)
    #plt.imshow(th)
    
    #Now pass this as a "Mask" to get the individual coins
    #and save them
    save_using_bb(iden, img, th)
    

    

def the_mask():
    training_dir = "../data/Masked/Labels/imgs/t"
    masked_dir = "../data/Masked/Labels/l"
    
    num_imgs=105
    
    for i in np.arange(num_imgs):
        img_name = training_dir+str(i)+".jpg"
        mask_name = masked_dir+str(i)+".png"
        ide = "img"+str(i)
        segment_using_masks(ide, img_name, mask_name)  

if __name__ == "__main__":
    #the_mask()
    #help_out_folders()