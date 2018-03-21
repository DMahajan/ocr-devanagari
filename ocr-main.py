import numpy as np
from sklearn import svm
from os.path import join
import skimage.io as sio
import os
import matplotlib.pyplot as plt
from skimage import filters
from skimage import transform
from skimage import feature
from sklearn.externals import joblib
import time
import sys



def getFeat(fileDir, featType, align):
	classListFile = 'classlist.txt'
	imgDim = 32

	classes = [line.strip() for line in open(classListFile,'r')]

	cell_size = 4
	block_size = 2
	n_orientations = 9
	if featType=="hog":
		n_block = imgDim//block_size
		n_cell = block_size
		featDim = (n_block**2)*(n_cell**2)*n_orientations
		featDim = 1764
	elif featType=="sum":
		featDim = 2*imgDim
	else:
		sys.exit("Did not enter feature type")
	#print "featDim: ",featDim
	xmat = np.zeros((1,featDim))
	ymat = np.zeros((1))
	for classNum, className in enumerate(classes):
		imgDir = trainDir + '/' + className + '/'
		print imgDir
		for imgFile in os.listdir(imgDir):
			img = sio.imread(imgDir + '/' + imgFile, True) # reads as greyscale
			#print img.shape
			if align:
				# Rotate to align characters according to Hough transform
				img_edge = feature.canny(img)
				h,theta,rho = transform.hough_line(img_edge)
				# find angles around pi/2 (which should correspond to shirorekha)
				h_peak, theta_peak, rho_peak = transform.hough_line_peaks(h,theta,rho,4)
				print theta_peak
				theta_rotate = np.abs(np.concatenate([theta_peak[theta_peak > 1.4], theta_peak[theta_peak < -1.4]]))
				print theta_rotate
				theta_rotate = np.mean(theta_rotate)
				theta_deg = (theta_rotate*180/np.pi) #counterclockwise rotation
				print "In degrees",theta_deg
				theta_deg = theta_deg - 90
				img_rot = transform.rotate(img,theta_deg)
				plt.imshow(img_edge,cmap='gray')
				fig = plt.figure()
				plt.subplot(1,2,1)
				plt.imshow(img,cmap='gray')
				plt.subplot(1,2,2)
				plt.imshow(img_rot,cmap='gray')
				plt.show()

			if featType == "hog":
				hogFeat = feature.hog(img,orientations=n_orientations,pixels_per_cell=(cell_size,cell_size),cells_per_block=(block_size,block_size))
				xmat = np.vstack((xmat,hogFeat))
			elif featType == "sum":
				val = filters.threshold_otsu(img)
				imgBin = img < val
				imgBin = imgBin.astype(bool)
				horSum = np.sum(imgBin,axis=0)
				vertSum = np.sum(imgBin,axis=1)
				combSum = np.hstack((horSum,vertSum))
				xmat = np.vstack((xmat,combSum))
			else:
				sys.exit("Did not enter feature type")

			ymat = np.vstack((ymat,classNum))

	xmat = xmat[1:,:]
	ymat = ymat[1:]
	print xmat.shape,ymat.shape
	np.save('xmat_'+featType+str(cell_size)+str(block_size)+str(n_orientations)+'.npy',xmat)
	np.save('ymat_'+featType+str(cell_size)+str(block_size)+str(n_orientations)+'.npy',ymat)
	print type(imgBin[0,0])

	return xmat,ymat

def train(modelFile,xmat,ymat):
	# Train SVM classifier
	clf = svm.SVC(C=2.0) #default: 'ovr', one-vs-rest instead of one-vs-one
	clf.fit(xmat, ymat)
	trainAccuracy = clf.score(xmat,ymat)# training error
	print trainAccuracy
	joblib.dump(clf, modelFile)


def test(modelFile,xmat,ymat):
	# SVM classifier
	clf = joblib.load(modelFile) 
	testAccuracy = clf.score(xmat,ymat)
	print testAccuracy


if __name__ == '__main__':
	start_time = time.time()
	
	featType = sys.argv[2]
	align = int(sys.argv[3]) #1 if yes, 0 if no
	modelFile = sys.argv[4]


	if sys.argv[1] == 'train':
		fileDir = 'DevanagariHandwrittenCharacterDataset/Train'
		xmat, ymat = getFeat(fileDir,featType,align)
		train(modelFile, xmat, ymat)
	elif sys.argv[1] == 'val':
		fileDir = 'DevanagariHandwrittenCharacterDataset/Val'
		xmat, ymat = getFeat(fileDir,featType,align)
		test(modelFile, xmat, ymat)
	else:
		fileDir = 'DevanagariHandwrittenCharacterDataset/Test'
		xmat, ymat = getFeat(fileDir,featType,align)
		test(modelFile, xmat, ymat)


	print("--- %s seconds ---" % (time.time() - start_time))
