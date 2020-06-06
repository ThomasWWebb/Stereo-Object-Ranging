import argparse
import numpy as np
import cv2
import math
import objectDistancing
from os import listdir
from os.path import isfile, join

useWLS = True #enables WLS filtering
useSparseStereo = False # enables sparse stereo instead of dense
master_path_to_dataset = "C:/Users/thoma/Documents/Third Year Completed/Vision/TTBB-durham-02-10-17-sub10/left-images/"
def drawPred(image, class_name, confidence, left, top, right, bottom, colour, distance):
    cv2.rectangle(image, (left, top), (right, bottom), colour, 1)

    label = '%s:%.2f' % (class_name, distance)
    #label = label + ' C:' + str(confidence)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

#####################################################################
def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    return (classIds_nms, confidences_nms, boxes_nms)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

confThreshold = 0.5 
nmsThreshold = 0.4  
inpWidth = 416  
inpHeight = 416     

classesFile = 'coco.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
output_layer_names = getOutputsNames(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

################################################################################

# define display window name
windowName = 'Object Ranging'
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

################################################################################

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

skipToFile = "1506942555.485817_L"
#skipToFile = ""
#applies object detection and ranging to each pair of images in the dataset

fileList = listdir(master_path_to_dataset)
for fileL in fileList:
    
    file = join(master_path_to_dataset, fileL)
    #skips to a specific file if one is set
    if len(skipToFile) != 0 and not (skipToFile in file):
        continue
    elif len(skipToFile) !=0 and skipToFile in file:
        skipToFile = ""  
    
    fullRightFile = (file.replace("_L", "_R")).replace("left", "right");
    fileR = fileL.replace("_L", "_R")

    if isfile(fullRightFile) and isfile(file):

        #read in the stereo image pair    
        frame = cv2.imread(file)
        imgR = cv2.imread(fullRightFile, cv2.IMREAD_COLOR)

        start_t = cv2.getTickCount()

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        # set the input to the CNN network
        net.setInput(tensor)
        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)
        # remove the bounding boxes with low confidence
        confThreshold = 0
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)
        
        #compute the disparity image of the stereo pair'
        if not useSparseStereo:
            if useWLS:
                disparityImg = objectDistancing.calculateDisparityWLS(frame, imgR)
            else:
                disparityImg = objectDistancing.calculateDisparity(frame, imgR)

        boxDistances = []

        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = left + width
            bottom = top + height

            distance = 0

            detectedObject = frame[top:top+height, left:right]
            
            #only estimate the distance of objects for which the majority of the box is within the disparity image
            if left + width // 2 >= 135 and top + height // 2 <= 390 and len(detectedObject) > 0:

                #calculate the distance of the object from the detection box
                if useSparseStereo:
                    distance = objectDistancing.calculateDisparitySparse(detectedObject,imgR[top:bottom, :], left)
                else:
                    distance = objectDistancing.calculateDepth(disparityImg, left,top, width, height)
                    cv2.imshow("disparity after", (disparityImg * (256. / 128)).astype(np.uint8));

                distance =  round(abs(distance), 3)
                boxDistances.append(distance)
            else:
                boxDistances.append(distance)

        minDistance = math.inf
        #display the estimations on the image, done after all distance estimations to prevent earlier estimation displays effecting later estimations in close proximity
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = left + width
            bottom = top + height
            
            if left + width // 2 >= 135 and top + height // 2 <= 390: #and boxDistances[detected_object] > 0:
                if minDistance > boxDistances[detected_object] and boxDistances[detected_object] > 0:
                    minDistance = boxDistances[detected_object]
                drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, right, bottom, (255, 178, 50), boxDistances[detected_object])


        if minDistance == math.inf:
            minDistance = 0
        print(fileL)
        print("{} : nearest detected scene object ({}m)".format(fileR, minDistance))

        #efficiency information
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        # display image
        cv2.imshow(windowName,frame)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, False )

        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        key = cv2.waitKey(5)

        # if user presses "x" then exit  / press "f" for fullscreen display
        if (key == ord('x')):
            break
        elif (key == ord('f')):
            args.fullscreen = not(args.fullscreen)

cv2.destroyAllWindows()


################################################################################


