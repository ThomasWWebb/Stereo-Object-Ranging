import cv2
import argparse
import math
import objectDistancing
import numpy as np
import glob
 
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

skipToFile = "1506942549.475784_L"

#applies object detection and ranging to each pair of images in the dataset
for file in glob.glob("C:/Users/thoma/Documents/Vision/TTBB-durham-02-10-17-sub10/left-images/*.png"):
    if len(skipToFile) != 0 and not (skipToFile in file):
        continue
    elif len(skipToFile) !=0 and skipToFile in file:
        skipToFile = ""
    #file = "C:/Users/thoma/Documents/Vision/TTBB-durham-02-10-17-sub10/left-images/1506942473.484027_L.png"
    frame = cv2.imread(file)

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
    
    #compute the disparity image of the stereo pair
    disparityImg = objectDistancing.calculateDisparityWLS(file, frame)

    boxDistances = []

    for detected_object in range(0, len(boxes)):
        box = boxes[detected_object]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height
        
        if left + width // 2 >= 135 and top + height // 2 <= 390:

            #calculate the distance of the object from the detection box
            distance = objectDistancing.calculateDepth(disparityImg, left,top, width, height)
            distance =  round(abs(distance), 3)
            boxDistances.append(distance)
            #drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, right, bottom, (255, 178, 50), distance)
        else:
            boxDistances.append(0)

    for detected_object in range(0, len(boxes)):
        box = boxes[detected_object]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height
        
        if left + width // 2 >= 135 and top + height // 2 <= 390:
            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, right, bottom, (255, 178, 50), boxDistances[detected_object])

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # display image
    cv2.imshow(windowName,frame)
    #cv2.imshow("disparity after", disparityImg);
    cv2.imshow("disparity after", (disparityImg * (256. / 64)).astype(np.uint8));
    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, False )

    # stop the timer and convert to ms. (to see how long processing and display takes)
    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

    # start the event loop + detect specific key strokes
    # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)
    key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

    # if user presses "x" then exit  / press "f" for fullscreen display
    if (key == ord('x')):
        break
    elif (key == ord('f')):
        args.fullscreen = not(args.fullscreen)

cv2.destroyAllWindows()


################################################################################


