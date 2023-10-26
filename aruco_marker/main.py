import cv2
import depthai as dai
import numpy as np

winName = "Aruco Marker Detection"

parameters = cv2.aruco.DetectorParameters()
parameters.minDistanceToBorder = 7
parameters.cornerRefinementMaxIterations = 149
parameters.minOtsuStdDev = 4.0
parameters.adaptiveThreshWinSizeMin = 7
parameters.adaptiveThreshWinSizeStep = 49
parameters.minMarkerDistanceRate = 0.014971725679291437
parameters.maxMarkerPerimeterRate = 10.075976700411534
parameters.minMarkerPerimeterRate = 0.2524866841549599
parameters.polygonalApproxAccuracyRate = 0.05562707541937206
parameters.cornerRefinementWinSize = 9
parameters.adaptiveThreshConstant = 9.0
parameters.adaptiveThreshWinSizeMax = 369
parameters.minCornerDistanceRate = 0.09167132584946237

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

detected_markers = {}
frame_counter = 0

# Create pipeline
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    videoQueue = device.getOutputQueue(name="video", maxSize=8, blocking=False)
    while True:
        frame_counter += 1
        try:
            inFrame = videoQueue.get()
            frame = inFrame.getCvFrame()

            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            if markerIds is not None:
                print('frame: {} ids: {}'.format(frame_counter, markerIds.tolist()))
            else:
                print('frame: {} ids: None'.format(frame_counter))

            im_out = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            cv2.imshow(winName, im_out)

            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print(e)

cv2.destroyAllWindows()