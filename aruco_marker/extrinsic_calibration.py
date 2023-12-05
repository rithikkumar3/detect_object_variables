import cv2
import numpy as np
import depthai as dai
from pathlib import Path

checkerboard_size = (4, 7) 
square_size = 0.0235  

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[1], 0:checkerboard_size[0]].T.reshape(-1, 2) * square_size

objpoints = []  
imgpoints = [] 

pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

with dai.Device(pipeline) as device:
    videoQueue = device.getOutputQueue(name="video", maxSize=8, blocking=False)

    while True:
        inFrame = videoQueue.get()
        frame = inFrame.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(frame, checkerboard_size, corners2, ret)
        
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) == ord('q'):
            break


retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


np.save('extrinsic_rvecs.npy', rvecs)
np.save('extrinsic_tvecs.npy', tvecs)

print("Extrinsic calibration is complete.")