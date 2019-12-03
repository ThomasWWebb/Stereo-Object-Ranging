import cv2
import math
import numpy as np

camera_focal_length_px = 399.9745178222656
camera_focal_length_m = 4.8 / 1000      
stereo_camera_baseline_m = 0.2090607502

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

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    disparitySum = 0
    pixelCount = 0

    left = left - 135
    if left < 0:
        left = 0
    
    box = disparity[top:top+height, left:left+width]

    disparityAvg = selectiveMean(box, height, width, 0.1, 0.1)
    #disparityAvg = np.nanmedian(box, None)
    #disparityAvg = centerPixel(box)
    #disparityAvg = gaussianSum(box, 1)

    if (disparityAvg > 0):

        Z = (f * B) / disparityAvg;
        return Z

    return 0;

##############################################################

def calculateDisparity(full_path_filename_left, imgL):
    #calculates the disparity image for a pair of stereo images
    
    max_disparity = 64;
                                            
    crop_disparity = True
    full_path_filename_right = (full_path_filename_left.replace("_L", "_R")).replace("left", "right");

    if ('.png' in full_path_filename_left):

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        #convert images to greyscale 
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
        
        #raise to power to improve calculation
        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        #to improve contrast
        #grayL = cv2.equalizeHist(grayL)
        #grayR = cv2.equalizeHist(grayR)

        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
        grayL = clahe.apply(grayL)
        grayR = clahe.apply(grayR)

        disparity = stereoProcessor.compute(grayL,grayR)
        
        #reduce noise
        dispNoiseFilter = 5;
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);

        cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));

        #crops unique sections of each camera
        if (crop_disparity):
            width = np.size(disparity_scaled, 1);
            disparity_scaled = disparity_scaled[0:390,135:width];

        return disparity_scaled

    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

#####################################################################

def calculateDisparityWLS(full_path_filename_left, imgL):
    #calculates the disparity image for a pair of stereo images combined with a wls filter
    
    max_disparity = 112;
    
    crop_disparity = True
    full_path_filename_right = (full_path_filename_left.replace("_L", "_R")).replace("left", "right");

    if ('.png' in full_path_filename_left):

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        #convert images to greyscale 
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
        
        #raise to power to improve calculation
        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        #to improve contrast
        #grayL = cv2.equalizeHist(grayL)
        #grayR = cv2.equalizeHist(grayR)

        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
        grayL = clahe.apply(grayL)
        grayR = clahe.apply(grayR)

        filterLambda = 80000
        sigma = 1.2
        
        leftSide = cv2.StereoSGBM_create(minDisparity = 0,
                                            numDisparities = max_disparity,
                                            blockSize = 21,
                                            P1 = 1,
                                            P2 = 5,
                                            disp12MaxDiff = 64,
                                            preFilterCap = 12,
                                            uniquenessRatio = 10,
                                            speckleWindowSize = 50,
                                            speckleRange = 2,
                                            mode = 0)
        rightSide = cv2.ximgproc.createRightMatcher(leftSide)
        wls = cv2.ximgproc.createDisparityWLSFilter(leftSide)
        wls.setLambda(filterLambda)
        wls.setSigmaColor(sigma)

        disparityL = leftSide.compute(grayL, grayR)

        cv2.imshow("disparity before", disparityL);
        disparityR = rightSide.compute(grayR, grayL)

        disparity = wls.filter(disparityL, grayL, None, disparityR)
        
        #reduce noise
        dispNoiseFilter = 5;
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);

        cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));

        #crops unique sections of each camera
        if (crop_disparity):
            width = np.size(disparity_scaled, 1);
            disparity_scaled = disparity_scaled[0:390,135:width];

        return disparity_scaled

    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();
