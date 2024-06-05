import sys
import math
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


##### Mouse Click Event #####

right_clicks = list()

def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global right_clicks
        right_clicks.append([x, y])
        print(right_clicks[-1])



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

# stereo = cv2.StereoBM_create(numDisparities=64, blockSize=19)
stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=19, P1 = 80, P2 = 1200)

# disparity = stereo.compute(imgL,imgR)
disparity = stereo.compute(left, right)
valid_pixels = disparity > 160.0

baseline = 50
fx  = 1.93
units = 0.6

depth = np.zeros(shape=left.shape).astype("float")
depth[valid_pixels] = (fx * baseline)/(units * disparity[valid_pixels])

depth_norm = np.uint8((depth - depth.min())/(depth.max() - depth.min())*255) 

cv2.imshow('depth', depth_norm)
cv2.waitKey(0)

dst = cv2.Canny(left, 5, 300, None, 3)
cv2.imshow('edges', dst)
cv2.waitKey(0)

cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)
# cdepth = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
cdepth = cv2.add(depth_norm, dst)
lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 2, 10)

# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(cdepth, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

# cv2.imshow("Source", left)
# cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdepth)
# cv2.waitKey(0)

cv2.namedWindow('Depth with edges', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Depth with edges', mouse_callback)
cv2.imshow('Depth with edges', cdepth)
cv2.waitKey(0)

if right_clicks:
    
    d1 = depth[right_clicks[0][1], right_clicks[0][0]]*1000
    d2 = depth[right_clicks[1][1], right_clicks[1][0]]*1000
    
    click1 = (240 - right_clicks[0][1])*depth[right_clicks[0][1], right_clicks[0][0]]
    click2 = (240 - right_clicks[1][1])*depth[right_clicks[1][1], right_clicks[1][0]]


    # print(right_clicks[0])
    # print(right_clicks[1])

    print(depth[int(right_clicks[0][1]), int(right_clicks[0][0])])
    print(depth[int(right_clicks[1][1]), int(right_clicks[1][0])])

    thickness_projection = abs(click1-click2)*(10/1.93)
    print(thickness_projection)
    thickness = math.sqrt(math.pow(abs(d2 - d1), 2) + math.pow(thickness_projection, 2))
    print(thickness)

plt.imshow(depth_norm,'inferno')
plt.show()