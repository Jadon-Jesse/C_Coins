# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 23:33:47 2017

@author: Jadon
"""
import os
import errno
import glob
import time
import cv2
import dlib
from skimage import io

def train_detector(save_name):
    #Point to folder
    folder = "../data/TrainHOG/"
    coins_folder = folder
    
    
    #Now train a simple object detector using dlib
    #Using the folowing options
    options = dlib.simple_object_detector_training_options()
    
    options.add_left_right_image_flips = True
    options.C = 10
    # Tell the code how many CPU cores your computer has for the fastest training.
    options.num_threads = 4
    options.be_verbose = True
    
    
    training_xml_path = os.path.join(coins_folder, "merged.xml")
    
    detective = os.path.join(coins_folder, str(save_name))
    dlib.train_simple_object_detector(training_xml_path, detective, options)
    print("done training")


def test_detector_accuracy(det_name):
    folder = "../data/TrainHOG/"
    coins_folder = folder
    training_xml_path = os.path.join(coins_folder, "traincoins.xml")
    testing_xml_path = os.path.join(coins_folder, "testcoins.xml")
    detective = os.path.join(coins_folder, str(det_name))
    print("")  # Print blank line to create gap from previous output
    print("Training accuracy: {}".format(
        dlib.test_simple_object_detector(training_xml_path, detective)))
    print("Testing accuracy: {}".format(
        dlib.test_simple_object_detector(testing_xml_path, detective)))


def run_detector(detector):
    
    folder = "../data/TrainHOG/"
    coins_folder = folder
    detective = os.path.join(coins_folder, str(detector))

    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector(detective)
    
    # We can look at the HOG filter we learned.  It should look like a coin.  Neat!
    win_det = dlib.image_window()
    win_det.set_image(detector)
    
    # Now let's run the detector over the images in the coins/Test folder and display the
    # results.
    test_fold = os.path.join(coins_folder,"ExData/" )
    print("Showing detections on the images in the test folder...")
    win = dlib.image_window()
    for f in glob.glob(os.path.join(test_fold, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)
        dets = detector(img)
        print("Number of coins detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
    
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        time.sleep(5)
        dlib.hit_enter_to_continue()
    
def make_dir(img_name):
    directory = "../data/Generated/detected/"+str(img_name)+"/"
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return directory

def crop_detections(detector):
    
    folder = "../data/TrainHOG/"
    coins_folder = folder
    detective = os.path.join(coins_folder, str(detector))

    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector(detective)
    
    # We can look at the HOG filter we learned.  It should look like a coin.  Neat!
    #win_det = dlib.image_window()
    #win_det.set_image(detector)
    
    # Now let's run the detector over the images in the coins/Test folder and display the
    # results.
    test_fold = "test/croptoclass/"#os.path.join(coins_folder,"ExData/" )
    print("Showing detections on the images in the test folder...")
    #win = dlib.image_window()
    i =0
    for f in glob.glob(os.path.join(test_fold, "*.jpg")):
        print("Processing file: {}".format(f))
        name = "img"+str(i)
        der=make_dir(name)
        img = cv2.imread(f)
        nj=name+".jpg"
        og = os.path.join(der, nj)
        cv2.imwrite(og, img)
        #io.imsave(og, img)
        dets = detector(img)
        print("Number of coins detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            crop_img = img[d.top():d.bottom(),d.left():d.right()]
            ide = str(k)+".jpg"
            place = os.path.join(der,ide)
            cv2.imwrite(place, crop_img)
            #io.imsave(place, crop_img)
        i+=1

if __name__ =="__main__":
    ##########################################################################
    #This Section trains the coindetector using the images data/TrainHOG/Train
    #And stores the resulting detector in data/TrainHOG/
    #
    ###########################################################################
    #detector_name = "../data/TrainHOG/coindetective2.svm"
    detector_name="real_coindetective.svm"
    #train_detector(detector_name)
    
    
    
    
    
    ###########################################################################
    #This Section Tests the allready train coin detector
    #
    #test_detector_accuracy(detector_name)
    #detector = dlib.simple_object_detector(detector_name)
    #win_det = dlib.image_window()
    #win_det.set_image(detector)

    ###########################################################################
    #This Section Creates more training imgs for the classifier
    #
    crop_detections(detector_name)
    
    












