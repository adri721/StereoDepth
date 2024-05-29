import pyrealsense2 as rs
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import datetime



##### Mouse Click Event #####

right_clicks = list()

def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global right_clicks
        right_clicks.append([x, y])
        print(right_clicks[-1])



ct = datetime.datetime.now()

foldername = str(ct.year)+"_"+str(ct.month)+"_"+str(ct.day)+"_"+str(ct.hour)+"_"+str(ct.minute)+"_"+str(ct.second)
print(foldername)
path = "test_images/"+foldername+"/"
os.mkdir("test_images/"+foldername)


pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.infrared)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    irL = frames.get_infrared_frame(1)
    irR = frames.get_infrared_frame(2)

    frameL =  np.asanyarray(irL.get_data())
    frameR =  np.asanyarray(irR.get_data())
    cv2.imshow('left', frameL)
    cv2.imshow('right', frameR)

    left = frameL
    right = frameR
    # imgL = cv2.resize(left, (0,0), None, 0.5, 0.5)
    # imgR = cv2.resize(right, (0, 0), None, 0.5, 0.5)

    # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=19)
    stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=13, P1 = 80, P2 = 1200)

    # disparity = stereo.compute(imgL,imgR)
    disparity = stereo.compute(left, right)
    global depth
    depth = (50*1.93)/(0.6*disparity)
    
    cv2.imshow('Depth', depth)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(path +"left.png", frameL)
        cv2.imwrite(path +"right.png", frameR)
        cv2.imwrite(path +"depth.png", depth)
        break

cv2.destroyAllWindows()


cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Depth', mouse_callback)
cv2.imshow('Depth', depth)
cv2.waitKey(0)

if right_clicks:
    click1 = (240 - right_clicks[0][1])*depth[right_clicks[0][1], right_clicks[0][0]]
    click2 = (240 - right_clicks[1][1])*depth[right_clicks[1][1], right_clicks[1][0]]

    # print(right_clicks[0])
    # print(right_clicks[1])

    print(depth[int(right_clicks[0][1]), int(right_clicks[0][0])])
    print(depth[int(right_clicks[1][1]), int(right_clicks[1][0])])

    difference = abs(click1-click2)*(10/1.93)
    print(difference)