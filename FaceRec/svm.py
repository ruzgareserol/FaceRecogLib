import cv2
import numpy as np
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import createHogOfAnImage

ruzgarHog = createHogOfAnImage.hogGeneration(r'/home/nvidia/Desktop/iSeeYou/ruzgar/')
print 'ruzgar tamam' + str(ruzgarHog.size)

orhunHog = createHogOfAnImage.hogGeneration(r'/home/nvidia/Desktop/iSeeYou/orhun/')
print 'orhun tamam' + str(orhunHog.size)

latifHog = createHogOfAnImage.hogGeneration(r'/home/nvidia/Desktop/iSeeYou/latif/')
print 'latif tamam' + str(latifHog.size)

mainMatrix = np.vstack(((ruzgarHog , orhunHog , latifHog )))
print np.shape(mainMatrix)
print mainMatrix


labels = [0 , 1 , 2]

print labels

clf=SVC(gamma=0.001,C=10)

clf.fit(mainMatrix,labels)

#ypred = clf.predict()
joblib.dump(clf, "hog.pkl", compress=3)





