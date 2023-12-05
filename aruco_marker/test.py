# import cv2
# import depthai as dai
# import numpy as np
# from utils import ARUCO_DICT

# winName = "Aruco Marker Detection"

# def pose_estimation(frame, aruco_dict_type, camera_matrix, distortion_coefficients):

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
#     parameters = cv2.aruco.DetectorParameters()

#     corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

#     # If markers are detected
#     if ids is not None:
#         # Estimate pose of each marker and return the values rvec and tvec
#         rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, distortion_coefficients)
        
#         for i in range(len(ids)):
#             # Draw a square around the markers
#             cv2.aruco.drawDetectedMarkers(frame, corners) 

#             # Draw Axis
#             cv2.aruco.drawAxis(frame, camera_matrix, distortion_coefficients, rvec[i], tvec[i], 0.1)  

#     return frame

# camera_matrix = np.load('calibration_matrix.npy')
# distortion_coefficients = np.load('distortion_coefficients.npy')


# parameters = cv2.aruco.DetectorParameters()
# parameters.minDistanceToBorder = 7
# parameters.cornerRefinementMaxIterations = 149
# parameters.minOtsuStdDev = 4.0
# parameters.adaptiveThreshWinSizeMin = 7
# parameters.adaptiveThreshWinSizeStep = 49
# parameters.minMarkerDistanceRate = 0.014971725679291437
# parameters.maxMarkerPerimeterRate = 10.075976700411534
# parameters.minMarkerPerimeterRate = 0.2524866841549599
# parameters.polygonalApproxAccuracyRate = 0.05562707541937206
# parameters.cornerRefinementWinSize = 9
# parameters.adaptiveThreshConstant = 9.0
# parameters.adaptiveThreshWinSizeMax = 369
# parameters.minCornerDistanceRate = 0.09167132584946237

# dictionary = cv2.aruco.DICT_5X5_50
# aruco_dict_type = ARUCO_DICT["DICT_5X5_50"]
# detected_markers = {}
# frame_counter = 0

# # Create pipeline
# pipeline = dai.Pipeline()
# cam = pipeline.createColorCamera()
# xout = pipeline.createXLinkOut()
# xout.setStreamName("video")
# cam.video.link(xout.input)

# # Connect to device and start pipeline
# with dai.Device(pipeline) as device:
#     videoQueue = device.getOutputQueue(name="video", maxSize=8, blocking=False)
#     while True:
#         frame_counter += 1
#         try:
#             inFrame = videoQueue.get()
#             frame = inFrame.getCvFrame()

#             markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
#             if markerIds is not None:
#                 print(f'frame: {frame_counter} ids: {markerIds.tolist()}')
#             else:
#                 print(f'frame: {frame_counter} ids: None')

#             im_out = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
#             im_out = pose_estimation(im_out, aruco_dict_type, camera_matrix, distortion_coefficients)
#             cv2.imshow(winName, im_out)

#             if cv2.waitKey(1) == ord('q'):
#                 break

#         except Exception as e:
#             print(e)

# cv2.destroyAllWindows()


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import depthai as dai


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Estimate pose of each marker and return the values rvec and tvec
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, matrix_coefficients, distortion_coefficients)
        
        # Draw a square around the markers
        cv2.aruco.drawDetectedMarkers(frame, corners)

        # Draw Axis and display ID for each marker
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvecs[i], tvecs[i], 0.01)
            
            # Display the ID of the marker
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'ID: {ids[i][0]}'
            position = tuple(corners[i][0][0].astype(int))
            cv2.putText(frame, text, position, font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return frame


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True)
    ap.add_argument("-d", "--D_Coeff", required=True)
    args = vars(ap.parse_args())

    aruco_dict_type = ARUCO_DICT["DICT_5X5_50"]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

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
            output = pose_estimation(frame, aruco_dict_type, k, d)
            cv2.imshow('Estimated Pose', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
