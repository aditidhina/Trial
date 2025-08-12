# Import necessary libraries
import cv2  # OpenCV for image processing, if need to install run pip install opencv-python in terminal and wait for it to finish
import os  # For working with files/folders
import numpy as np  # For numerical operations, if need to install run pip install numpy in terminal and wait for it to finish 
from ultralytics import YOLO  # For object detection, if need to install run pip install ultralytics in terminal and wait for it to finish

# if you want the images to show remove the hashes next to the cv.imshow and the waitkey

# Get the current working directory (where this script is located)
current_dir = os.getcwd()

# Set up folder paths:
# 'photos' folder is where we'll look for input images
input_dir = os.path.join(current_dir, 'photos')
# 'output' folder is where we'll save processed images
output_dir = os.path.join(current_dir, 'output')

# Create the output folder if it doesn't exist
# Note: You need to delete this folder each time before running to process new photos
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)  # Create the folder
    except OSError as e:
        print(f"Error creating directory: {e}")

# Load a pre-trained Haar Cascade classifier for detecting objects
# Note: This is actually a face detector (not birds) - you'd need a bird-specific one
bird_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the YOLOv8 object detection model (this is a powerful AI model)
# 'yolov8n.pt' is a small version good for general objects including birds
model = YOLO('yolov8n.pt')

# Get a list of all JPG images in the input folder
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg'))]

prev_img = None  # This will store the previous image for comparing changes

# Process each image one by one
for idx, image_file in enumerate(image_files):
    # Step 1: Load the image
    image_path = os.path.join(input_dir, image_file)
    bird_test = cv2.imread(image_path)
    
    # Check if the image loaded correctly
    if bird_test is None:
        print(f"Error loading image: {image_file}")
        continue  # Skip to next image if there's an error

    # Show the original image (commented out by default)
    cv2.imshow('Original Image', bird_test)

    # Convert image from BGR (OpenCV default) to RGB (what YOLO expects)
    bird_test_rgb = cv2.cvtColor(bird_test, cv2.COLOR_BGR2RGB)

    # --- YOLO Detection ---
    # Use YOLO to detect objects in the image with confidence threshold of 0.5 (50%)
    results = model(bird_test_rgb, conf=0.5)
    # Draw boxes around detected objects on the image
    annotated_img = results[0].plot()  

    # --- Change Detection ---
    # Only compare if we have a previous image (not the first image)
    if idx > 0 and prev_img is not None:
        # Convert both images to grayscale (easier to compare)
        gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(bird_test, cv2.COLOR_BGR2GRAY)
        
        # Find differences between current and previous image
        diff = cv2.absdiff(gray_prev, gray_current)
        # Convert differences to black/white (binary) image
        _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Find outlines (contours) of changed areas
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw boxes around significant changes (bigger than 100 pixels)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw yellow rectangle around changes
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # --- Bird Detection with Haar Cascade ---
    # Convert image to grayscale (needed for Haar Cascade)
    gray_img = cv2.cvtColor(bird_test, cv2.COLOR_BGR2GRAY)
    # Detect objects using Haar Cascade
    birds = bird_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    bird_detected = False
    # Draw blue rectangles around detected "birds" (actually faces with this classifier)
    for (x, y, w, h) in birds:
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        bird_detected = True

    # Print detection results
    if bird_detected:
        print(f"Bird detected in {image_file}!")
    elif idx > 0 and prev_img is not None and 'contours' in locals() and len(contours) > 0:
        print(f"Change detected in {image_file}, but no bird found.")

    # Show the processed image with all detections (commented out by default)
    cv2.imshow('Annotated Image', annotated_img)

    # Save the processed image to output folder
    # First convert back to BGR format (standard for OpenCV images)
    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"detected_{image_file}"), annotated_img)
    
    # Store current image for comparison with next image
    prev_img = bird_test.copy()
    
    # These lines are commented out - they would pause the program to show images
    # cv.waitKey(1)  
    # cv.waitKey(0)  

# Close all image windows when done
cv2.destroyAllWindows()   # This line is safe to keep even in a headless environment