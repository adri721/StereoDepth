import pyrealsense2 as rs
import cv2
import numpy as np
import os
import datetime
import serial
import time
import glob

new_calib_flag = True
recalib_flag = False

if new_calib_flag:
    ct = datetime.datetime.now()

    foldername = str(ct.year)+"_"+str(ct.month)+"_"+str(ct.day)+"_"+str(ct.hour)+"_"+str(ct.minute)+"_"+str(ct.second)
    print(foldername)

    path = foldername+"/"
    os.mkdir(foldername)
    os.mkdir(foldername+"/left")
    os.mkdir(foldername+"/right")


    image_counter = 0

    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.infrared)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    pipeline.start(config)


    while image_counter < 19:
        frames = pipeline.wait_for_frames()
        irL = frames.get_infrared_frame(1)
        irR = frames.get_infrared_frame(2)

        frameL =  np.asanyarray(irL.get_data())
        frameR =  np.asanyarray(irR.get_data())
        cv2.imshow('left', frameL)
        cv2.imshow('right', frameR)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(path +"left/"+ str(image_counter)+".png", frameL)
            cv2.imwrite(path +"right/"+ str(image_counter)+".png", frameR)
            image_counter+=1
            print(image_counter)

    cv2.destroyAllWindows()

if recalib_flag or new_calib_flag:
    if recalib_flag:
        foldername = "2024_5_14_15_23_37"
        print(foldername)

    path = foldername+"/"
    
    chessboardSize = (9, 7) 
    frameSize = (640,480) # Resolution of both cameras.

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = [] # 2d points in image plane.


    imagesLeft = glob.glob(path+'left/*.png')
    imagesRight = glob.glob(path+'right/*.png')

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):

        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:

            objpoints.append(objp)

            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv2.imshow('img left', imgL)
            cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv2.imshow('img right', imgR)
            cv2.waitKey(0)


    cv2.destroyAllWindows()

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same 

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    print(newCameraMatrixL)
    print(newCameraMatrixR)


    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

    print("Saving parameters!")
    cv_file = cv2.FileStorage(path+'stereoMap.xml', cv2.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])

    cv_file.release()


if recalib_flag == False and new_calib_flag == False:
    foldername = "2024_5_14_15_23_37"
    path = foldername + "/"

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open(path+'stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


def undistortRectify(frameR, frameL):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    return undistortedR, undistortedL


imagesLeft = glob.glob(path+'left/*.png')
imagesRight = glob.glob(path+'right/*.png')

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv2.imread(imgLeft)
    imgR = cv2.imread(imgRight)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    undistortedR, undistortedL = undistortRectify(grayR, grayL)
    cv2.imshow('undisR', undistortedR)
    cv2.imshow('undisL', undistortedL)
    cv2.waitKey(0)
