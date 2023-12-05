import cv2
import depthai as dai
import os


# Output directory for checkerboard images
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# Number of images to capture
num_images = 20

# Create pipeline
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.video.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    videoQueue = device.getOutputQueue(name="video", maxSize=8, blocking=False)
    
    image_count = 0
    while True:
        inFrame = videoQueue.get()
        frame = inFrame.getCvFrame()

        cv2.imshow('Capture Checkerboard Images', frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Save image to output directory
            image_filename = os.path.join(output_dir, f'image_{image_count:03d}.png')
            cv2.imwrite(image_filename, frame)
            print(f'Image saved as {image_filename}')
            image_count += 1
            
            if image_count >= num_images:
                break
        elif key == ord('q'):
            break

cv2.destroyAllWindows()
