#Robertas Dereskevicius 2019/11 University of Edinburgh
#Algorithmic Foundations of Data Science Assignment 1
from scipy.sparse import csr_matrix, identity
#from scipy import misc
#from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plot
from matplotlib.image import imread
from sklearn.preprocessing import normalize
import cv2
import numpy as np
import time
import math

#Resize and blur the image
def downsampleImage(image, sizeX, sizeY):
	modImage = cv2.resize(image,(sizeX,sizeY))
	modImage = cv2.GaussianBlur(modImage, (3,3), 0)
	#modImage = misc.imresize(image,(sizeX,sizeY))
	#modImage = gaussian_filter(modImage, 1)
	
	return modImage
	
#Compute adjacency matrix, diagonal matrix, sum of edges for each pixel
def constructAdjacencyMatrix(modImage, weightThreshold, edgeRadius):
	imgWidth = modImage.shape[0]
	imgHeight = modImage.shape[1]
	#Normalize img pixels
	modImage = np.true_divide(modImage, 255)
	totalSize = imgWidth * imgHeight
	rowID = []
	colID = []
	contents = []
	numEdges = 0
	sumArr = np.zeros((totalSize))
	
	newPixels = np.zeros((totalSize + edgeRadius * 2 , totalSize + edgeRadius * 2, 5))
	
	#Create a new normalized image that has boundaries which repeat from the other side and have small distance
	iHeight = -edgeRadius
	maxHeight = imgHeight + edgeRadius * 2
	maxWidth = imgWidth + edgeRadius * 2
	normDistH = 1 / (imgHeight - 1)
	normDistW = 1 / (imgWidth - 1)
	while iHeight < maxHeight:
		realHeight = iHeight + edgeRadius
		transformHeight = iHeight
		if transformHeight >= imgHeight:
			transformHeight -= imgHeight
		iWidth = -edgeRadius
		while iWidth < maxWidth:
			realWidth = iWidth + edgeRadius
			transformWidth = iWidth
			if transformWidth >= imgWidth:
				transformWidth -= imgWidth
			newPixels[realHeight][realWidth][0] = modImage[transformHeight][transformWidth][0]
			newPixels[realHeight][realWidth][1] = modImage[transformHeight][transformWidth][1]
			newPixels[realHeight][realWidth][2] = modImage[transformHeight][transformWidth][2]
			newPixels[realHeight][realWidth][3] = normDistH * iHeight
			newPixels[realHeight][realWidth][4] = normDistW * iWidth
			iWidth += 1
		iHeight += 1
	
	#Pixeloccupancy saves us time by avoiding pixels which were fully parsed already
	pixelOccupancy = np.zeros((imgHeight, imgWidth))
	#Main loop
	for i in range(imgHeight):
		curMultiple = i * imgHeight
		for j in range(imgWidth):
			curIndex = curMultiple + j
			#Vector of R,G,B,x,y
			curVec = newPixels[i + edgeRadius][j + edgeRadius]
			
			pixelOccupancy[i,j] = 1
			
			topLeft = [i, j]
			bottomRight = [i + 1 + edgeRadius * 2, j + 1 + edgeRadius * 2]
			
			#Calculate weights for the whole pixelRadius by pixelRadius window (faster)
			destSlice = newPixels[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]
			destSlice = np.exp(-4 * np.power(np.linalg.norm(np.subtract(destSlice, curVec), axis=2), 2))
			
			iHeight = -edgeRadius + i
			itHeight = 0
			maxMat = edgeRadius * 2 + 1

			#Iterate through each window
			while itHeight < maxMat:
				realHeight = iHeight + itHeight
				if realHeight < 0:
					realHeight = imgHeight + realHeight
				elif realHeight >= imgHeight:
					realHeight = realHeight - imgHeight
					
				iWidth = -edgeRadius + j
				itWidth = 0
				while itWidth < maxMat:
					realWidth = iWidth + itWidth
					if realWidth < 0:
						realWidth = imgWidth + realWidth
					elif realWidth >= imgWidth:
						realWidth = realWidth - imgWidth
						
					if pixelOccupancy[realHeight,realWidth] == 0:
						edgeWeight = destSlice[itHeight, itWidth]
						#If window pixel weight is above threshold
						if edgeWeight >= weightThreshold:
							destIndex = realHeight * imgHeight + realWidth
							#Data for the original edge
							rowID.append(curIndex)
							colID.append(destIndex)
							contents.append(edgeWeight)
							sumArr[curIndex] += edgeWeight

							#Symmetrical edge from the other side as graph is undirected
							rowID.append(destIndex)
							colID.append(curIndex)
							contents.append(edgeWeight)
							sumArr[destIndex] += edgeWeight
							numEdges += 1
					itWidth += 1
				itHeight += 1
	#As we add indices for sparse matrix to arrays, we create a csr_matrix using them and their values
	adj = csr_matrix((contents, (rowID, colID)), shape=(totalSize, totalSize), dtype=np.float64)
	diagRow = []
	diagContents = []
	#Create diagonal matrix with sum of weights for each pixel at diag
	for i in range(totalSize):
		diagRow.append(i)
		diagContents.append(sumArr[i])
	diagCol = diagRow
	#Diag matrix is sparse too
	diag = csr_matrix((diagContents, (diagRow, diagCol)), shape=(totalSize, totalSize), dtype=np.float64)
	#degree = 2 * number of edges / number of pixels (in undirected graphs)
	avgDegree = 2 * numEdges / totalSize

	return adj, numEdges, avgDegree, diag, sumArr
	
#Power method with no initial vector for smallest eigenvector
def computePowerMethod(M,k,n):
	#randVec = np.random.uniform(low=-1, high=1, size=(n,))
	randVec = np.random.normal(0, 1, n)
	prevVec = randVec
	i = 1
	while i <= k:
		newVec = M * prevVec
		newVec_norm = np.linalg.norm(newVec)
		prevVec = newVec / newVec_norm
		i += 1
	return prevVec
	
#Power method with an initial value used to compute second smallest eigenvector
def computeSecondPowerMethod(M,k,firstVector,n):
	#randVec = np.random.uniform(low=-1, high=1, size=(n,))
	randVec = np.random.normal(0, 1, n)
	prevVec = randVec - firstVector * np.dot(randVec, firstVector)
	i = 1
	while i <= k:
		newVec = M * prevVec
		newVec_norm = np.linalg.norm(newVec)
		prevVec = newVec / newVec_norm
		i += 1
	return prevVec
	
#Main method that creates a laplacian matrix and uses it to compute second smallest eigenvector
def computeSecondEigenValue(adj, k, maxSize, diagonal):
	diagonal.eliminate_zeros()
	#D^-1/2
	powerDiagonal = diagonal.power(-(1/2))
	#Symmetric laplacian computed using the equation in the notes
	symLaplacian = identity(maxSize) - (powerDiagonal * adj * powerDiagonal)
	#Matrix M computed which can be used for the smallest eigenvector computation
	M = 2 * identity(maxSize) - symLaplacian
	#Smallest eigenvector computed
	smallestEigenVec = computePowerMethod(M,k,maxSize)
	#Second smallest eigenvector computed
	secondSmallestVec = computeSecondPowerMethod(M, k, smallestEigenVec, maxSize)
	return secondSmallestVec

#Function that evaluates each set of pixels using formula found in question 4 (sparse cut)
def evaluateSet(S, sumArr, maxSize, adjArr, prevS, 
		newValue, adj, prevNotS, prevVolS, prevVolNotS):
		
	eval = 0	
	notS = []
	topEval = 0
	volS = 0
	volNotS = 0
	#If its the first pixel in a set
	if prevS == 0:
		allIndexes = np.arange(maxSize)
		topEval = sumArr[newValue]
		notS = np.setdiff1d(allIndexes, S, assume_unique=True)
		volS = sumArr[S].sum()
		volNotS = sumArr[notS].sum()
	else:
		notS = np.setdiff1d(prevNotS, [newValue], assume_unique=True)
		
		#Used throughout so sum does not have to be recomputed
		newSum = sumArr[newValue]
		volS = prevVolS + newSum
		volNotS = prevVolNotS - newSum
		
		topEval = prevS
		connectionValue = 0
		
		shouldSum = True
		#Optimization as we only need to compute one sum of connections and can subtract the other from totalSize
		#Additionally, the smaller set is chosen to process
		if len(S) > len(notS):
			#Fastest way to access pixels that need updating
			rows, cols = (adj[newValue, :]).nonzero()
			uniqueCols = np.setdiff1d(cols, notS, assume_unique=True)
			
			#Array access is faster than csr_matrix
			for it2 in uniqueCols:
				connectionValue += adjArr[newValue][it2]
			shouldSum = False
		else:
			#Fastest way to access pixels that need updating
			rows, cols = (adj[newValue, :]).nonzero()
			uniqueCols = np.setdiff1d(cols, S, assume_unique=True)
			
			for it2 in uniqueCols:
				connectionValue += adjArr[newValue][it2]
		
		#Optimiza
		if shouldSum == True:
			topEval += connectionValue
			topEval -= (sumArr[newValue] - connectionValue)
		else:
			topEval -= connectionValue
			topEval += (sumArr[newValue] - connectionValue)
		
	botEval = 0
	if volNotS < volS:
		botEval = volNotS
	else:
		botEval = volS
		
	#Edge case with a pixel not having any weighted connections
	if botEval <= 0:
		botEval = 1e-8
	
	return (topEval / botEval), topEval, notS, volS, volNotS
	
#Finds two sparse cuts in an image
def findSparseCut(secondEigen, adj, sumArr, maxSize):
	sortedVertices = np.argsort(secondEigen)
	t = 0
	S = []
	prevNotS = []
	S_ = [sortedVertices[0]]
	adjArr = adj.toarray()
	topS = 0
	unscaledTopS = 0
	prevVolS = 0
	prevVolNotS = 0
	topS_, placeholder, pl2, pl3, pl4 = evaluateSet(S_, sumArr, maxSize, adjArr, 0, sortedVertices[0], adj, [], 0, 0)
	#Edge case with the first pixel having 0 weight sum
	if math.isnan(topS_) or topS_ != 1:
		topS_ = 999999
	while t < (len(secondEigen)-2):
		t += 1
		S.append(sortedVertices[t])
		#Main computation of the equation from question 4 to evaluate set S, we don't need to
		#compute S_ as its saved
		topS, unscaledTopS, prevNotS, prevVolS, prevVolNotS = evaluateSet(S, sumArr, maxSize,
			adjArr, unscaledTopS, sortedVertices[t], adj, prevNotS, prevVolS, prevVolNotS)
		if topS < topS_:
			#Must copy a np array as simply setting it points it to the same array
			#which can then be modified outside, not copying results in one sparse cut being only 1 px
			S_ = np.copy(S)
			topS_ = topS
	return S_

fileName = input("Insert the processed file name: ")
image = []
success = True
k = 1500
try:
	image = cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2RGB)
	#image=misc.imread(fileName)
	print("Image has been loaded...")
except:
	print("Could not find the specified file name for image " + fileName)
	success = False

if success == True:
	tic = time.clock()
	imSize = [100,100]
	maxSize = imSize[0] * imSize[1]
	weightLimit = 0.9
	pixelRadius = 5

	modImage = downsampleImage(image, imSize[0], imSize[1])
	figure = plot.figure()
	figure.suptitle('Algorithm results!')
	blurredImg = figure.add_subplot(131)
	eigenVec = figure.add_subplot(132)
	sparseCut = figure.add_subplot(133)
	prev = time.clock()
	#Get adj matrix, diag matrix, sum of weights for each pixel
	adjacencyMatrix, numEdges, avgDegree, diagonalMatrix, weightsSum = constructAdjacencyMatrix(modImage, 
		weightLimit, pixelRadius)
	print("Adjacency&diagonal computed! Time: " + str(time.clock() - prev))
	prev = time.clock()
	print("Number of edges: " + str(numEdges))
	print("Average degree: " + str(avgDegree))

	#Get second smallest eigenvector
	print("Chosen k value: " + str(k))
	secondSmallestEigenVec = computeSecondEigenValue(adjacencyMatrix, 
								k, maxSize, diagonalMatrix)
	print("Second smallest eigen computed!: Time: " + str(time.clock() - prev))
	prev = time.clock()
	#Find sparse cut
	cutResult = findSparseCut(secondSmallestEigenVec, adjacencyMatrix, weightsSum, maxSize)
	print("Sparse cut found! Time: " + str(time.clock() - prev))
	cutImage = np.zeros((maxSize))
	for it in cutResult:
		cutImage[it] = 1
	eigenImg = np.reshape(secondSmallestEigenVec, (imSize[0], imSize[1]))
	cutImage = np.reshape(cutImage, (imSize[0], imSize[1]))
	blurredImg.imshow(modImage)
	eigenVec.imshow(eigenImg)
	sparseCut.imshow(cutImage)
	toc = time.clock()
	print("Total Processing time: " + str(toc-tic))
	plot.show()

