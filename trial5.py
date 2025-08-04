

import cv2
import os
import numpy as np
from ultralytics import YOLO


# if you want the images to show remove the hashes next to the cv.imshow and the waitkey


# Input and output folders
current_dir = os.getcwd()

# sets the input directory to the current one
input_dir = os.path.join(current_dir, 'photos')

# sets the output directory
output_dir = os.path.join(current_dir, 'output')

# creates the output folder
# but you have to delete the folder every time you want to re input the photos
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        print(f"Error creating directory: {e}")

# You can find bird-specific cascades online or train your own
bird_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLOv8 model (use 'yolov8n.pt' for birds or custom-trained IR model) and if you haven't installed it yet, use `pip install ultralytics`
model = YOLO('yolov8n.pt')

# Takes the images from the photos folder, just in case there was another document in it
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg'))]

prev_img = None  # For change detection

for idx, image_file in enumerate(image_files):
    # reads the image(s)
    image_path = os.path.join(input_dir, image_file)
    bird_test = cv2.imread(image_path)
    
    # to check if the image was loaded successfully
    if bird_test is None:
        print(f"Error loading image: {image_file}")
        continue

    cv2.imshow('Original Image', bird_test)

    # Convert to RGB for YOLO
    bird_test_rgb = cv2.cvtColor(bird_test, cv2.COLOR_BGR2RGB)

    # --- YOLO Detection ---
    results = model(bird_test_rgb, conf=0.5)
    annotated_img = results[0].plot()  # Draws bounding boxes

    # --- Change Detection ---
    if idx > 0 and prev_img is not None:
        gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(bird_test, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_prev, gray_current)
        _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # Filter small changes
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Yellow for changes

    # --- Bird Detection with Haar Cascade ---
    gray_img = cv2.cvtColor(bird_test, cv2.COLOR_BGR2GRAY)
    birds = bird_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    bird_detected = False
    for (x, y, w, h) in birds:
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for birds
        bird_detected = True

    if bird_detected:
        print(f"Bird detected in {image_file}!")
    elif idx > 0 and prev_img is not None and 'contours' in locals() and len(contours) > 0:
        print(f"Change detected in {image_file}, but no bird found.")

    cv2.imshow('Annotated Image', annotated_img)

    # Save the annotated image
    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(os.path.join(output_dir, f"detected_{image_file}"), annotated_img)
    
    # Update previous image for next iteration
    prev_img = bird_test.copy()
    
    # Saves to output folder

    # cv.waitKey(1)  

# cv.waitKey(0)  
cv2.destroyAllWindows()   # This line is safe to keep even in a headless environment