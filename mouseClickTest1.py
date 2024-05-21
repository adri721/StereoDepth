import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import datetime



files = os.listdir("test_images")
for i in range(len(files)):
    print(str(i)+". "+str(files[i]))
print("Enter index of the folder")
folder_number = input()
foldername = str(files[int(folder_number)])
print(foldername)
path = "test_images/"+foldername+"/"


##### Mouse Click Event #####

right_clicks = list()

def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global right_clicks
        right_clicks.append([x, y])
        print(right_clicks[-1])


disp = cv2.imread(path +"disp.png", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Input image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Input image', mouse_callback)
cv2.imshow('Input image', disp)
cv2.waitKey(0)