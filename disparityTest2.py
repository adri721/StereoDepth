import pyrealsense2 as rs
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import datetime
import serial
import time
import glob


##### Mouse Click Event #####

right_clicks = list()

def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global right_clicks
        right_clicks.append([x, y])
        print(right_clicks[-1])


new_data_flag = False

if new_data_flag:
    ct = datetime.datetime.now()

    foldername = str(ct.year)+"_"+str(ct.month)+"_"+str(ct.day)+"_"+str(ct.hour)+"_"+str(ct.minute)+"_"+str(ct.second)
    print(foldername)
    path = "test_images/"+foldername+"/"
    os.mkdir("test_images/"+foldername)



    image_counter = 0

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(path +"left.png", frameL)
            cv2.imwrite(path +"right.png", frameR)
            break

    cv2.destroyAllWindows()

else:
    #user input?
    files = os.listdir("test_images")
    for i in range(len(files)):
        print(str(i)+". "+str(files[i]))
    print("Enter index of the folder")
    folder_number = input()
    foldername = str(files[int(folder_number)])
    print(foldername)
    path = "test_images/"+foldername+"/"


left = cv2.imread(path +"left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread(path +"right.png",cv2.IMREAD_GRAYSCALE)

imgL = cv2.resize(left, (0,0), None, 0.5, 0.5)
imgR = cv2.resize(right, (0, 0), None, 0.5, 0.5)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=19)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()

depth = (50*1.93)/(0.6*disparity)

cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Depth', mouse_callback)
cv2.imshow('Depth', depth)
cv2.waitKey(0)

click1 = right_clicks[0][1]*depth[right_clicks[0][1],right_clicks[0][0]]
click2 = right_clicks[1][1]*depth[right_clicks[1][1], right_clicks[1][0]]

# print(right_clicks[0])
# print(right_clicks[1])

print(depth[int(right_clicks[0][1]), int(right_clicks[0][0])])
print(depth[int(right_clicks[1][1]), int(right_clicks[1][0])])

difference = abs(click1-click2)*(12/1.93)
print(difference)

plt.imshow(depth,'inferno')
plt.show()