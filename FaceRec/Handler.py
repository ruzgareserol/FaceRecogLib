#this class, as can be seen from it's name is the handler of the whole project
#it handles the packages, naming, image processing, cropping and even training the svm classifier

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import numpy
import os
import glob
import cv2
import sys
import takeImagesViaCam
import createHogOfAnImage
import resize
import renameFilesAsIndexes
import cropNsave

#first of all, we have to take images via cam.
pplCount = 0
labels = numpy.array([])
names = numpy.array([])
nameLabels = numpy.array([])
print 'gathering and processing data from database'

for filename in os.listdir('/home/nvidia/Desktop/FacesOfPeople/'):
    pplCount = pplCount+1
    nameLabels = numpy.append(nameLabels , int(pplCount))
    names = numpy.append(names , str(filename))
    HM = createHogOfAnImage.hogGeneration(filename)
    if pplCount <= 1:
        FM = HM
    else:
        FM = numpy.vstack((FM, HM))
    for x in range(0, 200):
        labels = numpy.append(labels, int(pplCount))

print (names)
print'Names and corresponding labels are:'
print (nameLabels)
print 'currently we have ' + str(pplCount) + ' people'
#print'HOG matrice:'
#print(HM)



while True:
    print 'you can exit the input stage if you input "done"  '
    name = raw_input('input a person name =')
    print 'People count is '+ str(pplCount)
    if name == 'done':
        print 'finished importing people.'
        break
    else:
        pplCount = pplCount + 1
        names = numpy.append(names, str(name))
        takeImagesViaCam.takeImagesViaCam(name)
        print 'finished capturing'
        # from this point, we have images of people, saved in imageData/theirName
        # now, we are going to crop their faces via haar_cascade of cv2
        # the cropped images will be saved in a seperate folder called iSeeYou/theirName
        #they will also be resized to 450x450 images
        cropNsave.cropAndSaveThemIntoSpecificFaceFolders(0, name)
        print 'finished cropping and saving in the folder: iSeeYou'
        renameFilesAsIndexes.renameAsIndexes(name)
        print 'images are now renamed as indexes and saving in the folder: facesOfPeople'
        # from this point, we have faces cropped and resized for our purpose of HOG generation
        # imageData contains cam outputs
        # iSeeYou contains cropped-resized images
        # facesOfPeople contains the data we need for HOG generation
        # let's begin generating HOGs for multiclass svm classification and prediction
        HM = createHogOfAnImage.hogGeneration(name  )
        print 'succesfully created a feature submatrix for ' + name
        if pplCount <=1:
            FM = HM
        else:
            FM = numpy.vstack((FM , HM))
        print name + 'is added to feature matrix'
        nameLabels = numpy.append(nameLabels , pplCount)
        print 'feature submatrix:'
        print (FM)
        for x in range(0,200):
            labels = numpy.append(labels, int(pplCount))



print'Names and corresponding labels are' \
     '' \
     ''
print (names)
print (nameLabels)
print 'currently we have ' + str(pplCount) + ' people'

file2write = open('/home/nvidia/Desktop/ImageData/dataBase/names', 'w')
file2write.write(str(names))
file2write.close()

file2write2 = open('/home/nvidia/Desktop/ImageData/dataBase/nameLabels', 'w')
file2write2.write(str(names))
file2write2.close()

#finished constructing the featres matrix and labels
#now it is time to construct the classifier
print 'constructing the classifier'
clf = SVC(gamma=0.001 , C=10)
clf.fit(FM, labels)

#let's complete the construction of the classifier
#from now the classifier will be trained

joblib.dump(clf , "FaceRecognizer.pk1" ,compress= 4)
print 'Classifier ready'







