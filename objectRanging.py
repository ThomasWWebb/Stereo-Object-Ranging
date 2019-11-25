import cv2
import argparse
import sys
import math
import numpy as np
import glob

camera_focal_length_px = 399.9745178222656
camera_focal_length_m = 4.8 / 1000      
stereo_camera_baseline_m = 0.2090607502     

image_centre_h = 262.0;
image_centre_w = 474.5;

def project_disparity_to_3d(disparity, left, top, width, height):

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    disparitySum = 0
    pixelCount = 0

    if top + height > len(disparity):
        height = len(disparity) - top
    if left + width > len(disparity[0]):
        width = len(disparity[0]) - left
        

    for yStepper in range(0, height):
        for xStepper in range(0, width):
            disparitySum += disparity[top + yStepper, left + xStepper]
            pixelCount += 1

    disparityAvg = disparitySum / pixelCount

    if (disparityAvg > 0):

        Z = (f * B) / disparityAvg;
        return Z

    return 0;

def calculateDisparity(full_path_filename_left):

    max_disparity = 128;
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);
    crop_disparity = False
    full_path_filename_right = (full_path_filename_left.replace("_L", "_R")).replace("left", "right");

    print(full_path_filename_left);
    print(full_path_filename_right);

    if ('.png' in full_path_filename_left):

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(grayL,grayR);

        # filter out noise and speckles (adjust parameters as needed)!!!!

        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);


        if (crop_disparity):
            width = np.size(disparity_scaled, 1);
            disparity_scaled = disparity_scaled[0:390,135:width];

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        image = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)

        cv2.imshow("disparity", image);

        return disparity_scaled

    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

################################################################################

keep_processing = True

# parse command line arguments for camera ID or video file, and YOLO files
parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-fs", "--fullscreen", action='store_true', help="run in full screen mode")
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')

args = parser.parse_args()

################################################################################
# dummy on trackbar callback function
def on_trackbar(val):
    return

#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, confidence, left, top, right, bottom, colour, distance):
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    label = '%s:%.2f' % (class_name, confidence)
    label = label + ' D:' + str(distance)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

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

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

################################################################################

# define display window name + trackbar

windowName = 'YOLOv3 object detection: ' + args.weights_file
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowName , 0, 100, on_trackbar)

################################################################################

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

for file in glob.glob("C:/Users/thoma/Documents/Vision/TTBB-durham-02-10-17-sub10/left-images/*.png"):
    #file = "C:/Users/thoma/Documents/Vision/TTBB-durham-02-10-17-sub10/right-images/1506943061.478682_R.png"
    image = cv2.imread(file)

    # start a timer (to see how long processing and display takes)
    start_t = cv2.getTickCount()

    frame = image

    # rescale if specified
    if (args.rescale != 1.0):
        frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

    # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
    tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # set the input to the CNN network
    net.setInput(tensor)

    # runs forward inference to get output of the final output layers
    results = net.forward(output_layer_names)

    # remove the bounding boxes with low confidence
    confThreshold = cv2.getTrackbarPos(trackbarName,windowName) / 100
    classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

    disparityImg = calculateDisparity(file)

    for detected_object in range(0, len(boxes)):
        box = boxes[detected_object]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height
        
        depth = project_disparity_to_3d(disparityImg, left,top, width, height)
        distance =  round(abs(depth), 3)
        drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, right, bottom, (255, 178, 50), distance)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # display image
    cv2.imshow(windowName,frame)
    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN & args.fullscreen)

    # stop the timer and convert to ms. (to see how long processing and display takes)
    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

    # start the event loop + detect specific key strokes
    # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)
    key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

    # if user presses "x" then exit  / press "f" for fullscreen display
    if (key == ord('x')):
        keep_processing = False
    elif (key == ord('f')):
        args.fullscreen = not(args.fullscreen)



# close all windows
#cv2.destroyAllWindows()


################################################################################


