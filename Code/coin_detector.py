# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 23:33:47 2017

@author: Jadon
"""
import os
import glob
import time

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
    options.C = 5
    # Tell the code how many CPU cores your computer has for the fastest training.
    options.num_threads = 4
    options.be_verbose = True
    
    
    training_xml_path = os.path.join(coins_folder, "traincoins.xml")
    
    detective = os.path.join(coins_folder, str(save_name))
    print(detective)
    print(training_xml_path)
    dlib.train_simple_object_detector(training_xml_path, "coindetective.svm", options)
    print("done training")


def test_detector_accuracy(det_name):
    folder = "../data/TrainHOG/"
    coins_folder = folder
    training_xml_path = os.path.join(coins_folder, "traincoins.xml")
    testing_xml_path = os.path.join(coins_folder, "testcoins.xml")
    print("")  # Print blank line to create gap from previous output
    print("Training accuracy: {}".format(
        dlib.test_simple_object_detector(training_xml_path, "coindetective.svm")))
    print("Testing accuracy: {}".format(
        dlib.test_simple_object_detector(testing_xml_path, "coindetective.svm")))


def run_detector():
    
    folder = "../data/TrainHOG/"
    coins_folder = folder


    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector("coindetective.svm")
    
    # We can look at the HOG filter we learned.  It should look like a coin.  Neat!
    win_det = dlib.image_window()
    win_det.set_image(detector)
    
    # Now let's run the detector over the images in the coins/Test folder and display the
    # results.
    test_fold = os.path.join(coins_folder,"Test/" )
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
    





if __name__ =="__main__":
    
    detector_name = "coindetective.svm"
    train_detector(detector_name)



















