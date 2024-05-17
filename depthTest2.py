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

left = cv2.resize(left, (0,0), None, 0.5, 0.5)
right = cv2.resize(right, (0, 0), None, 0.5, 0.5)

fx = 1.93  # 50  # 942.8  # lense focal length
baseline = 50.0  # distance in mm between the two cameras
disparities = 128  # num of disparities to consider
block = 13  # block size to match
units = 0.512  # depth units, adjusted for the output to fit in one byte
sbm = cv2.StereoBM_create(numDisparities=disparities,
                      blockSize=block)
left_matcher = cv2.StereoBM_create(numDisparities=disparities, blockSize=block)
wlsFilter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
disparityL = left_matcher.compute(left, right)
disparityR = right_matcher.compute(left, right)

sigma = 1.5
lmbda = 3200.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher);
wls_filter.setLambda(lmbda);
wls_filter.setSigmaColor(sigma);

filtered_disp = wls_filter.filter(disparityL, left, disparity_map_right=disparityR);

# calculate disparities
disparity = sbm.compute(left, right)
numpy_horizontal = np.hstack((left, right))
hori = np.hstack((disparityL, filtered_disp))
cv2.imshow('HorizontalStack1', numpy_horizontal)
cv2.imshow('HoriStack2', hori)
cv2.waitKey(0)
valid_pixels = disparity > 0

# calculate depth data
depth = np.zeros(shape=left.shape).astype("uint8")
depth[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])

# visualize depth data
depth = cv2.equalizeHist(depth)
colorized_depth = np.zeros((left.shape[0], left.shape[1], 3), dtype="uint8")
temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
colorized_depth[valid_pixels] = temp[valid_pixels]
plt.imshow(colorized_depth)
plt.show()