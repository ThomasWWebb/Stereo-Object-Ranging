import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

camera_focal_length_px = 399.9745178222656
camera_focal_length_m = 4.8 / 1000      
stereo_camera_baseline_m = 0.2090607502

max_disparity = 128

crop_disparity = True

def selectiveMedian(disparityBox, height, width, detractHeight, detractWidth):
    return np.nanmedian(disparityBox[math.floor(height*detractHeight):math.floor(height*(1 - detractHeight)),
                                   math.floor(width*detractWidth):math.floor(width*(1 - detractWidth))])

#################################################################


def selectiveMean(disparityBox, height, width, detractHeight, detractWidth):
    return np.nanmean(disparityBox[math.floor(height*detractHeight):math.floor(height*(1 - detractHeight)),
                                   math.floor(width*detractWidth):math.floor(width*(1 - detractWidth))])

#################################################################

def centerPixel(disparityBox):
    return (disparityBox[len(disparityBox) // 2, len(disparityBox[0]) // 2])
    

################################################################

def gaussianSum(disparityBox, sigma):

    maskWidth = min(len(disparityBox)//2, len(disparityBox[0])//2)
    mask = getMask(maskWidth, sigma)
    disparitySum = 0
    yStart = len(disparityBox) // 2 - maskWidth // 2
    xStart = len(disparityBox[0]) // 2 - maskWidth // 2

    for yIndex in range(0, maskWidth):
        for xIndex in range(0, maskWidth):
            disparitySum = disparitySum + disparityBox[yStart + yIndex, xStart + xIndex] * mask[yIndex][xIndex]
            
    return disparitySum


def getMask(size, s):
    mask = np.zeros([size, size], dtype = float)
    centre = int(size / 2)
    shift = size // 2
    for i in range(0, size):
        for j in range(0, size):
            iDist = abs(centre - i)
            jDist = abs(centre - j)
            mask[i][j] = getGauss(s, iDist, jDist);
    maskSum = np.sum(mask)
    mask = mask / maskSum
    return mask;

def getGauss(s, x, y):
    gauss = (1/((2*math.pi)*s**2))*math.e**(-((y**2)+(x**2))/(2*s**2))
    return(gauss);

####################################################################
    
def calculateDepth(disparity, left, top, width, height):

    disparitySum = 0
    pixelCount = 0

    left = left - 135
    if left < 0:
        left = 0
    
    box = disparity[top:top+height, left:left+width]

    #disparityAvg = selectiveMean(box, height, width, 0.125, 0.125)
    #disparityAvg = selectiveMedian(box, height, width, 0.125, 0.125)
    #disparityAvg = centerPixel(box)
    disparityAvg = gaussianSum(box, 1)

    if (disparityAvg > 0):

        depth = (camera_focal_length_px * stereo_camera_baseline_m) / disparityAvg;
        return depth

    return 0

##############################################################

def calculateDisparity(imgL, imgR):
    #calculates the disparity image for a pair of stereo images

    stereoProcessor = cv2.StereoSGBM_create(minDisparity = 0,
                                            numDisparities = max_disparity,
                                            blockSize = 21)

    #convert images to greyscale 
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    #to reduce the impact of variable illumination 
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)

    #grayL = cv2.equalizeHist(grayL)
    #grayR = cv2.equalizeHist(grayR)

    #raise to power to improve calculation
    grayL = np.power(grayL, 0.75).astype('uint8')
    grayR = np.power(grayR, 0.75).astype('uint8')

    disparity = stereoProcessor.compute(grayL,grayR)
    
    #reduce noise
    dispNoiseFilter = 5;
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);

    #crops unique sections of each camera
    if (crop_disparity):
        width = np.size(disparity_scaled, 1);
        disparity_scaled = disparity_scaled[0:390,135:width];

    return disparity_scaled

#####################################################################

def calculateDisparityWLS(imgL, imgR):
    #calculates the disparity image for a pair of stereo images combined with a wls filter
    #utilising the online tutorial https://docs.opencv.org/master/d3/d14/tutorial_ximgproc_disparity_filtering.html

    #convert images to greyscale 
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)
    
    #raise to power to improve calculation
    grayL = np.power(grayL, 0.75).astype('uint8');
    grayR = np.power(grayR, 0.75).astype('uint8');

    #to improve contrast
    #grayL = cv2.equalizeHist(grayL)
    #grayR = cv2.equalizeHist(grayR)

    filterLambda = 80000
    sigma = 1.2

    #creates both matchers
    leftSide = cv2.StereoSGBM_create(minDisparity = 0,
                                        numDisparities = max_disparity,
                                        blockSize = 11)
    rightSide = cv2.ximgproc.createRightMatcher(leftSide)
    #initialises the wls filter
    wls = cv2.ximgproc.createDisparityWLSFilter(leftSide)
    wls.setLambda(filterLambda)
    wls.setSigmaColor(sigma)

    #computes both disparity images
    disparityL = leftSide.compute(grayL, grayR)
    disparityR = rightSide.compute(grayR, grayL)

    #applies speckle filtering to both disparity images
    dispNoiseFilter = 5;
    cv2.filterSpeckles(disparityL, 0, 4000, max_disparity - dispNoiseFilter);
    cv2.filterSpeckles(disparityR, 0, 4000, max_disparity - dispNoiseFilter);
    
    #thresholds and scales for display
    _, disparityLScaled = cv2.threshold(disparityL,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparityLShow = (disparityLScaled / 16).astype(np.uint8)

    #applies the wls filter to both disparity images to produce one image
    disparity = wls.filter(disparityL, grayL, None, disparityR)
    
    #further speckle filtering to the final image, thresholding and scaling for distance calculation
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);
    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16).astype(np.uint8);

    #crops unique sections of each camera
    if (crop_disparity):
        width = np.size(disparity_scaled, 1);
        disparity_scaled = disparity_scaled[0:390,135:width];

    return disparity_scaled

##############################################################################

def calculateDisparitySparse(queryImage, searchImage, left):
    #calculates the disparity of a detected object using sparse stereo
    #utilising your example and the online tutorials https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
    
    #Adds a border to enable feature point matching
    queryImage = cv2.copyMakeBorder(queryImage, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0)
    searchImage = cv2.copyMakeBorder(searchImage, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0);

    #convert images to greyscale 
    queryImage = cv2.cvtColor(queryImage,cv2.COLOR_BGR2GRAY);
    searchImage = cv2.cvtColor(searchImage,cv2.COLOR_BGR2GRAY);

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    #queryImage = clahe.apply(queryImage)
    #searchImage = clahe.apply(searchImage)

    # Create detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with ORB
    queryKeypoints, queryDescriptors = orb.detectAndCompute(queryImage,None)
    searchKeypoints, searchDescriptors = orb.detectAndCompute(searchImage,None)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6,
                       key_size = 12,   
                       multi_probe_level = 1)

    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(queryDescriptors,searchDescriptors,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(0, len(matches))]
    good = []
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.7*n.distance:
                good.append(m)
        except ValueError:
            pass
    queryKeypointsCoords = []
    searchKeypointsCoords = []

    for mat in good:
        # Get the matching keypoints for each of the images
        searchImageIndex = mat.trainIdx
        queryImageIndex = mat.queryIdx

        #Get the coordinates of the matches
        (x1,y1) = queryKeypoints[queryImageIndex].pt
        (x2,y2) = searchKeypoints[searchImageIndex].pt
        if y2 == y1:
            queryKeypointsCoords.append(x1)
            searchKeypointsCoords.append(x2)

    if not len(queryKeypointsCoords):
        return 0
    
    return distanceFromCoords(queryKeypointsCoords,searchKeypointsCoords, left)

def distanceFromCoords(queryKeypointsCoords,searchKeypointsCoords, left):
    disparitySum = 0
    for index in range(0, len(queryKeypointsCoords)):
        disparitySum += abs(queryKeypointsCoords[index] + left - searchKeypointsCoords[index])

    disparityAvg = disparitySum / len(queryKeypointsCoords)
    if (disparityAvg > 0):

        depth = (camera_focal_length_px * stereo_camera_baseline_m) / disparityAvg;
        return depth

    return 0
    
    
