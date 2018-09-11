import cv2
import numpy as np
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib


def hogGeneration(name):
    personPath = r'/home/nvidia/Desktop/FacesOfPeople/' + name
    path, dirs, files = next(os.walk(personPath))
    hog = cv2.HOGDescriptor()
    #hm = hog matrice
    hm = np.array([])
    #the images after 50 shoots are considered because it takes some time for the camera to get used to the lighting of the shooting environment
    for x in range(50 ,  250 ):
        filename = personPath + '/' + str(x-1) + '.jpeg'
        im = cv2.imread(filename)
        h = hog.compute(im, winStride=(450, 450), padding=(0, 0))  # storing HOG features as column vector
        h_trans = h.transpose()  # transposing the column vector
        if x == 50:
            hm = h_trans
        else:
            hm = np.vstack((hm, h_trans))  # appending it to the array
    return hm
