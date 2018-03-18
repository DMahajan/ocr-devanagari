import numpy as np
from sklearn import svm
from os.path import join
import skimage.io as sio
import os
import matplotlib.pyplot as plt
from skimage import filters
from sklearn.externals import joblib
import time
import sys



def get_feat():
	trainDir = '/home/dsm/Documents/CS231A/project/DevanagariHandwrittenCharacterDataset/Train'
	classListFile = 'classlist.txt'
	imgDim = 32

	classes = [line.strip() for line in open(classListFile,'r')]
	classNum = 0
	#xmat = np.zeros((1,imgDim**2),dtype=bool)
	xmat = np.zeros((1,2*imgDim))
	ymat = np.zeros((1))
	for className in classes:
		imgDir = trainDir + '/' + className + '/'
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

def train():
	# Train SVM classifier
	clf = svm.SVC() #default: 'ovr', one-vs-rest instead of one-vs-one
	clf.fit(xmat, ymat)
	trainAccuracy = clf.score(xmat,ymat)# training error
	print trainAccuracy
	joblib.dump(clf, 'svm1.pkl') #CHANGE FILE NAME EACH TIME



if __name__ == '__main__':
	start_time = time.time()

	get_feat()
	train()

	print("--- %s seconds ---" % (time.time() - start_time))
