import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('left1_Infrared.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('right1_Infrared.png', cv2.IMREAD_GRAYSCALE)

imgL = cv2.resize(imgL, (0,0), None, 0.5, 0.5)
imgR = cv2.resize(imgR, (0, 0), None, 0.5, 0.5)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()