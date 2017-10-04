# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:56:14 2017

@author: Jadon
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, errno
import shutil
import glob

def hough_to_mask_circles(image_name):
  
    
    img = image_name
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # detect circles
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=40, minRadius=5, maxRadius=50)
    circles = np.uint16(np.around(circles))
    
    # draw mask
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only 
    for i in circles[0, :]:
        cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
    
    # get first masked value (foreground)
    #fg = cv2.bitwise_or(img, img, mask=mask)
    
    # get second masked value (background) mask must be inverted
    #mask = cv2.bitwise_not(mask)
    #background = np.full(img.shape, 255, dtype=np.uint8)
    #bk = cv2.bitwise_or(background, background, mask=mask)
    
    # combine foreground+background
    #final = cv2.bitwise_or(fg, bk)
    plt.imshow(mask)
    return img,mask

#helper function to add seperate dirs for each type of coin in each 
#image folder so to make things easier


def copy_rename(src_file, dst_dir,old_name, new_file_name):
        shutil.copy(src_file,dst_dir)
        
        os.rename(old_name, new_file_name)

def help_out_sortcoins():
    dir_tar = "../data/Generated/TrainClassifier/"
    dir_src = "../data/Generated/detected/"
    coin_dirs = ["10C/","20C/","50C/","1R/", "2R/", "5R/", "Huh/"]
    coin_names = ["10C_","20C_","50C_","1R_", "2R_", "5R_", "Huh_"]
    #for each img folder
    for i in np.arange(178):
        #go into each subfolder
        for j in np.arange(len(coin_dirs)):
            folder_i = dir_src+"img"+str(i)+"/"+coin_dirs[j]+"*"
            dst_i = dir_tar+coin_dirs[j]

            coins_i = glob.glob(folder_i)
            #print(folder_i)
            #for each coin in the image folder
            for k in np.arange(len(coins_i)):
                c_split = coins_i[k].split("\\")
                nm = c_split[-1]
                old_dst_name = dst_i+ str(nm)
                new_dst_name = dst_i+coin_names[j]+str(i)+"_"+str(k)+".jpg"
                copy_rename(coins_i[k], dst_i,old_dst_name, new_dst_name)
            
def help_out_folders():
    directory = "../data/Generated/detected/"
    coin_dirs = ["10C/","20C/","50C/","1R/", "2R/", "5R/", "Huh/"]
    for i in np.arange(178):
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
    #d = make_dir(i)
    img, contours, heirh = cv2.findContours(masked_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #copy og img to folder for reference
    #h = d+"og.jpg"
    #cv2.imwrite(h, og_img)
    idx =0 
    for i in np.arange(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        roi=og_img[y:y+h,x:x+w]
        place =str(idx)
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
        
def process_img(image_name):
    image = cv2.imread(image_name)
    img, mask = hough_to_mask_circles(image)
    
    #save_using_bb(1,img,mask)
    
    return mask


#resize and cobine training imgs for the classifier
def resize_and_combine():
    dirs = ["../data/Generated/coins/", "../data/Generated/TrainClassifier/"] 
    dir_tar =["../data/Generated/Resized/"]
    coin_dirs = ["10C/","20C/","50C/","1R/", "2R/", "5R/", "Huh/"]
    coin_names=["10C_","20C_","50C_","1R_", "2R_", "5R_", "Huh_"]
    
    #for each dir that has imgs we want
    for n in np.arange(len(dirs)):
        #go through each subdir
        
        for i in np.arange(len(coin_dirs)):
            #copy and resize each coin into the new dir
            
            f = dirs[n]+coin_dirs[i]+"*.jpg"
            #get a list of all the files
            files = glob.glob(f)
            #for each file copy and rezise
            for j in np.arange(len(files)):
                img = cv2.imread(files[j])
                resized_image = cv2.resize(img, (100, 100))
                new_path = dir_tar[0]+coin_dirs[i] +coin_names[i]+ str(n)+"_"+str(j)+".jpg"
                cv2.imwrite(new_path, resized_image)
                
                
#create more training images by augmenting
def augment():
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    datagen = ImageDataGenerator(
        rotation_range=359,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.2,
        fill_mode='nearest')

    dir_tar ="../data/Generated/Resized/"
    coin_dirs = ["10C/","20C/","50C/","1R/", "2R/", "5R/"]
    #We want each coin to have around 2000 imgs
    coin_multip=[9, 5,3,50, 8, 4]
    
    for i in np.arange(len(coin_dirs)):
        fold=foldall = dir_tar+coin_dirs[i]
        foldall = dir_tar+coin_dirs[i]+"*.jpg"
        files = glob.glob(foldall)
        for f in np.arange(len(files)):
            
            img = load_img(files[f])  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            
            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            n = 0
            pre = "aug"+str(f)
            for batch in datagen.flow(x, batch_size=1, save_to_dir=fold, save_prefix=pre, save_format="jpg"):
                n += 1
                if n >= int(coin_multip[i]):
                    break  # otherwise the generator would loop indefinitely
            
if __name__ == "__main__":
    #the_mask()
    #help_out_folders()
    #help_out_sortcoins()
    #test_img = "test/t15.jpg"
    #img = cv2.imread(test_img)
    #g =process_img(test_img)
    #plt.imshow(g)
    #resize_and_combine()
    augment()