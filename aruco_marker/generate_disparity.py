import cv2
import numpy as np
import os

def compute_disparity_sgbm(left_img_path, right_img_path, disparity_output_path):
    # Read the left and right images
    left_image = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    # Set up the stereo matcher
    window_size = 5
    min_disp = 0
    num_disp = 160 - min_disp
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the disparity map
    print("Computing the disparity map...")
    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

    # Normalize the disparity for display
    disparity_normalized = cv2.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # Save the disparity map
    cv2.imwrite(disparity_output_path, disparity_normalized)
    print(f"Disparity map saved to {disparity_output_path}")

# Define the paths for left and right images
base_dir = "/home/rithik/Desktop/bsdr/scripts/frames_new"
left_dir = os.path.join(base_dir, "rect_left")
right_dir = os.path.join(base_dir, "rect_right")
disparity_dir = os.path.join(base_dir, "disparity")

# Create the output directory if it doesn't exist
if not os.path.exists(disparity_dir):
    os.makedirs(disparity_dir)

# Assuming the images are named sequentially as 'new_rect_left_0.png', 'new_rect_left_1.png', ...
# Adjust the range as per the number of image pairs available
for i in range(0, 10):  # Replace 10 with the actual number of image pairs
    left_image_path = os.path.join(left_dir, f"new_rect_left_{i}.png")
    right_image_path = os.path.join(right_dir, f"new_rect_right_{i}.png")
    disparity_output_path = os.path.join(disparity_dir, f"disparity_{i}.png")

    if os.path.exists(left_image_path) and os.path.exists(right_image_path):
        compute_disparity_sgbm(left_image_path, right_image_path, disparity_output_path)
    else:
        print(f"Image pair {i} not found.")
