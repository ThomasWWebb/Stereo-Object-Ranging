# Autonomous Vehicles: Using Stereo Vision for Object Distance Ranging

A solution for correctly detecting pedestrians and vehicles within the scene in-front of a vehicle and estimates the range (distance in metres) to those objects. Provided with You Only Look Once - YOLO for object detection this implementation operates upon a provided dataset of stereo pairs of images from video sequences taken from a vehicle driving through Durham city. 

Sparse and dense stereo ranging was used through different approaches to object matching and object disparity calculation. A coloured polygon highlights the object location in the scene and is annotated with a distance estimate. For each image pair the solution also displays to standard output the distance to the closest identified object.


