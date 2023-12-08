import numpy as np


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

print("O_T_EE\n", O_T_EE.transpose())

# cam_to_obj = np.eye(4)
# cam_to_obj[:3, :3] = R_final
# cam_to_obj[:3, 3] = tvec_final

cam_to_obj = np.array([[ 0.81554741, -0.02072131, -0.57831916,  0.05404661],
    [-0.04092182, -0.99892196, -0.02191647,  0.00542969],
    [-0.57724157,  0.04153979, -0.81551617,  1.0638568 ],
    [ 0.,          0.,          0.,          1.        ]])

# rob_to_cam = O_T_EE.transpose()@np.linalg.inv(cam_to_obj)
rob_to_cam = O_T_EE.transpose()@cam_to_obj

print(rob_to_cam)


robot_to_tag = np.eye(4)
robot_to_tag[:3,3]= [5,0,0]
camera_to_tag = np.eye(4)
camera_to_tag[:3,3] = [0,-5,0]

robot_to_camera = robot_to_tag@np.linalg.inv(camera_to_tag)
print(robot_to_camera)