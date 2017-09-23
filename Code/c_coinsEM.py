# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:56:14 2017

@author: Jadon
"""

import numpy as np
import cv2
import matplotlib.pyplot as pl

def getsamples(img):
    x, y, z = img.shape
    samples = np.empty([x * y, z])
    index = 0
    for i in range(x):
        for j in range(y):
            samples[index] = img[i, j]
            index += 1
    return samples


def EMSegmentation(img, no_of_clusters=2):
    output = img.copy()
    colors = np.array([[0, 11, 111], [22, 22, 22]])
    samples = getsamples(img)
    em = cv2.ml.EM_create()
    em.setClustersNumber(no_of_clusters)
    em.trainEM(samples)
    means = em.getMeans()
    covs = em.getCovs()  # Known bug: https://github.com/opencv/opencv/pull/4232
    x, y, z = img.shape
    distance = [0] * no_of_clusters
    for i in range(x):
        for j in range(y):
            for k in range(no_of_clusters):
                diff = img[i, j] - means[k]
                distance[k] = abs(np.dot(np.dot(diff, covs[k]), diff.T))
            output[i][j] = colors[distance.index(max(distance))]
    return output




def im2vec(I):
    """
    converts a 2D image + 1D for HSV/RGB to 1D row vector+ 1D for HSV/RGB
    """
    I = I.copy()
    rows, cols, dims = I.shape
    x = np.reshape(I, ( rows * cols, dims))
    return x

def vec2im(x, rows, cols, dims):
    """
    converts a 1D row vector back to og image
    """
    x = x.copy()
    I = np.reshape(x, (rows, cols, dims))
    return I

def normalize_image(I):
    """
    normalizes input image for displaying
    """
    I = I.copy()
    I = I - np.min(I)
    I = I / np.max(I)
    return I

def plot_vector(x, rows, cols, name='', title=''):
    """
    plots a vector x as 2D image4
    """
    I = x.copy()
    I = vec2im(I, rows, cols)
    I = normalize_image(I)
    pl.figure()
    pl.imshow(I, cmap=pl.cm.gray)
    pl.xticks([])
    pl.yticks([])
    pl.title(title, fontsize=16)
    if name != '':
        pl.savefig(name, bbox_inches='tight', dpi=300)

og_row = 960
og_cols = 1280
og_dims = 3
training_dir = "../CoinTrainingSet/Training/"

pic = training_dir+"t58.jpg"

bg = "../CoinTrainingSet/Training/t5.jpg"

#img = cv2.imread(pic)
#img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#I=im2vec(img)
#i2 = vec2im(I, og_row, og_cols, og_dims)

#pl.imshow(i2)

img = cv2.imread(pic)
pl.imshow('image', img)
pl.imshow('EM', output)
