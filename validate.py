import numpy as np
from sklearn import svm
from os.path import join
import skimage.io as sio
import os
import matplotlib.pyplot as plt
from skimage import filters
from sklearn.externals import joblib
import time

start_time = time.time()

testDir = '/home/dsm/Documents/CS231A/project/DevanagariHandwrittenCharacterDataset/Test'
classListFile = 'classlist.txt'
imgDim = 32
modelFile = 'svm1.pkl'


# Load and extract features
classes = [line.strip() for line in open(classListFile,'r')]
classNum = 0
#xmat = np.zeros((1,imgDim**2),dtype=bool)
xmat = np.zeros((1,2*imgDim))
ymat = np.zeros((1))
for className in classes:
	imgDir = testDir + '/' + className + '/'
	print imgDir
	for imgFile in os.listdir(imgDir):
		img = sio.imread(imgDir + '/' + imgFile, True) # reads as greyscale
		val = filters.threshold_otsu(img)
		imgBin = img < val
		imgBin = imgBin.astype(bool)
		#imgSum = np.sum(imgBin)
		horSum = np.sum(imgBin,axis=0)
		vertSum = np.sum(imgBin,axis=1)
		#imgVec = imgBin.flatten()# vectorize
		combSum = np.hstack((horSum,vertSum))
		xmat = np.vstack((xmat,combSum))
		ymat = np.vstack((ymat,classNum))
	classNum += 1
xmat = xmat[1:,:]
ymat = ymat[1:]
print xmat.shape,ymat.shape
print type(imgBin[0,0])
#plt.imshow(imgBin, cmap='gray')
#plt.show()

# SVM classifier
clf = joblib.load(modelFile) 
testAccuracy = clf.score(xmat,ymat)
print testAccuracy


print("--- %s seconds ---" % (time.time() - start_time))
