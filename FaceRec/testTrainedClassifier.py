import cv2
from sklearn.externals import joblib
import numpy
import os
import cropNsave
import resize

def predictionFunction(facePath):
    # crop faces and do some resizing and renaming

    # load the classifier
    clf = joblib.load('/home/nvidia/PycharmProjects/FaceRec/FaceRecognizer.pk1')

    img = cv2.imread(facePath)

    hog = cv2.HOGDescriptor()
    h = hog.compute(img, winStride=(450, 450), padding=(0, 0))
    h = h.transpose()

    nameFolder = open('/home/nvidia/Desktop/ImageData/dataBase/names', 'r')

    namesArray = nameFolder.read().split("' '")
    nameFolder.close()

    prdct = clf.predict(h[0].reshape(1, -1))
    print prdct
    print 'This person is predicted to be ' + str(namesArray[int(prdct) - 1])

