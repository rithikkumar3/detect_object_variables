import numpy as np
import cv2
import sys
import time
import json
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
from utils import ARUCO_DICT
import depthai as dai
from mpl_toolkits.mplot3d import Axes3D

R_list = []
def get_euler_angles(rvec):
    # Convert rotation vector to rotation matrix
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec, R)
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    R_list.append(R)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    tvecs_list = []
    rvecs_list = []
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.045, matrix_coefficients, distortion_coefficients)
        for rvec, tvec, corner, id_num in zip(rvecs, tvecs, corners, ids):
            cv2.aruco.drawDetectedMarkers(frame, [corner])
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            cv2.putText(frame, str(id_num[0]), tuple(corner[0][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            tvecs_list.append(tvec[0])
            rvecs_list.append(rvecs[0])
    return frame, tvecs_list, rvecs_list

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to the calibration matrix .npy file")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to the distortion coefficients .npy file")
    args = vars(ap.parse_args())

    aruco_dict_type = ARUCO_DICT["DICT_5X5_50"]
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])

    # Initialize pipeline
    pipeline = dai.Pipeline()

    # Create color camera
    cam = pipeline.createColorCamera()
    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.video.link(xout.input)

    # Initialize lists to store data
    frame_times = []
    x_coords = []
    y_coords = []
    z_coords = []

    roll_list = []
    pitch_list = []
    yaw_list = []

    tvecs_list_mat = []

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        videoQueue = device.getOutputQueue(name="video", maxSize=8, blocking=False)

        frame_count = 0
        while True:
            inFrame = videoQueue.get()
            frame = inFrame.getCvFrame()

            start_time = time.time()
            output, tvecs_list, rvecs_list = pose_estimation(frame, aruco_dict_type, k, d) 
            tvecs_list_mat = tvecs_list
            end_time = time.time()
            for rvec in rvecs_list:
                euler_angles = get_euler_angles(rvec)
                roll_list.append(euler_angles[0])
                pitch_list.append(euler_angles[1])
                yaw_list.append(euler_angles[2])

            frame_times.append(end_time - start_time)
            for tvec in tvecs_list:
                x_coords.append(tvec[0])
                y_coords.append(tvec[1])
                z_coords.append(tvec[2])

            cv2.imshow('Estimated Pose', output)
            if cv2.waitKey(1) == ord('q'):
                break

            frame_count += 1

    cv2.destroyAllWindows()

    coordinate_data = {
        'x_coordinates': x_coords,
        'y_coordinates': y_coords,
        'z_coordinates': z_coords
    }

    # Write frame processing times to a JSON file
    with open('frame_time_deltas.json', 'w') as f:
        json.dump(frame_times, f)
    
    with open('coordinate_changes.json', 'w') as f:
        json.dump(coordinate_data, f, indent=4)

    # Plot and save the time elapsed vs frames graph
    # plt.plot(frame_times)
    # plt.xlabel('Frames')
    # plt.ylabel('Time Delta (s)')
    # plt.title('Time Elapsed vs Frames')
    # plt.savefig('time_elapsed_vs_frames.png')
    # # plt.show()

    # # Plot and save the x, y, z coordinates vs frames graphs
    # plt.figure(figsize=(12, 8))
    # plt.subplot(3, 1, 1)
    # plt.plot(x_coords, label='X Coordinate')
    # plt.legend(loc='upper right')
    # plt.ylabel('X Coordinate (m)')

    # plt.subplot(3, 1, 2)
    # plt.plot(y_coords, label='Y Coordinate')
    # plt.legend(loc='upper right')
    # plt.ylabel('Y Coordinate (m)')

    # plt.subplot(3, 1, 3)
    # plt.plot(z_coords, label='Z Coordinate')
    # plt.legend(loc='upper right')
    # plt.xlabel('Frames')
    # plt.ylabel('Z Coordinate (m)')

    # plt.suptitle('Marker Translation Coordinates vs Frames')
    # plt.savefig('translation_coordinates_vs_frames.png')
    # # plt.show()

    # plt.figure(figsize=(12, 8))
    # ax = plt.axes(projection='3d')  # Create a new 3D axes

    # # Data for a three-dimensional scatter
    # ax.scatter3D(x_coords, y_coords, z_coords, c=z_coords, cmap='YlGn')

    # ax.set_xlabel('X Coordinate (m)')
    # ax.set_ylabel('Y Coordinate (m)')
    # ax.set_zlabel('Z Coordinate (m)')
    # plt.title('3D Scatter Plot of ArUco Marker Movement')

    # # Save the figure
    # plt.savefig('3d_scatter_plot.png')

    # Show the plot
    # plt.show()

    orientation_data = {
        'roll': roll_list,
        'pitch': pitch_list,
        'yaw': yaw_list
    }

    # Write orientation data to a JSON file
    with open('orientation_changes.json', 'w') as f:
        json.dump(orientation_data, f, indent=4)

    # Plot and save the orientation changes vs frames graphs
    # plt.figure(figsize=(12, 8))
    # plt.subplot(3, 1, 1)
    # plt.plot(roll_list, label='Roll')
    # plt.legend(loc='upper right')
    # plt.ylabel('Roll (rad)')

    # plt.subplot(3, 1, 2)
    # plt.plot(pitch_list, label='Pitch')
    # plt.legend(loc='upper right')
    # plt.ylabel('Pitch (rad)')

    # plt.subplot(3, 1, 3)
    # plt.plot(yaw_list, label='Yaw')
    # plt.legend(loc='upper right')
    # plt.xlabel('Frames')
    # plt.ylabel('Yaw (rad)')

    # plt.suptitle('Marker Orientation Changes vs Frames')
    # plt.savefig('orientation_changes_vs_frames.png')
    # plt.show()

    np.save('R_lists.npy',R_list)

    # R_size = len(R_list)
    # R_final = R_list[R_size-1]

    R_final = np.array([[ 0.81554741, -0.02072131, -0.57831916],
        [-0.04092182, -0.99892196, -0.02191647],
        [-0.57724157,  0.04153979, -0.81551617]])

    # tvecs_size = len(tvecs_list_mat)
    # tvec_final = tvecs_list_mat[tvecs_size-1]
    tvec_final = np.array([0.05404661, 0.00542969, 1.0638568 ])


    O_T_EE = np.array([[ 0.0630684, -0.0270113, -0.997634 ,  0.       ],
       [-0.997311 , -0.0388318, -0.0619966,  0.       ],
       [-0.037066 ,  0.998881 , -0.0293883,  0.       ],
       [ 0.364411 ,  0.357783 ,  0.274576 ,  1.       ]])

    # cam_to_obj = np.eye(4)
    # cam_to_obj[:3, :3] = R_final
    # cam_to_obj[:3, 3] = tvec_final

    cam_to_obj = np.array([[ 0.81554741, -0.02072131, -0.57831916,  0.05404661],
        [-0.04092182, -0.99892196, -0.02191647,  0.00542969],
        [-0.57724157,  0.04153979, -0.81551617,  1.0638568 ],
        [ 0.,          0.,          0.,          1.        ]])
    
    rob_to_cam = O_T_EE@np.linalg.inv(cam_to_obj)
    print(rob_to_cam)