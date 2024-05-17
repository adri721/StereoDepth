import pyrealsense2 as rs
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import datetime
import serial
import time
import glob


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

left = cv2.imread(path +"left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread(path +"right.png",cv2.IMREAD_GRAYSCALE)

imgL = cv2.resize(left, (0,0), None, 0.5, 0.5)
imgR = cv2.resize(right, (0, 0), None, 0.5, 0.5)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()