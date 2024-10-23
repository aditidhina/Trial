import cv2 as cv
import os

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

# Takes the images from the photos folder, just in case there was another document in it
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg'))]

for image_file in image_files:
    # reads the image(s)
    image_path = os.path.join(input_dir, image_file)
    bird_test = cv.imread(image_path)

    # to check if the image was loaded successfully
    if bird_test is None:
        print(f"Error loading image: {image_file}")
        continue

    # cv.imshow('Original Image', bird_test)

    # Grayscale
    sosad_blacknwhite = cv.cvtColor(bird_test, cv.COLOR_BGR2GRAY)
    # cv.imshow('Grayscale Image', sosad_blacknwhite)

    # Blurred
    blurry_wurry = cv.GaussianBlur(bird_test, (67, 67), cv.BORDER_DEFAULT)
    # cv.imshow('Blurred Image', blurry_wurry)

    # Edges
    edges = cv.Canny(bird_test, 35, 45)
    # cv.imshow('Edges', edges)

    # Dilate
    dialate_bird = cv.dilate(edges, (11, 11), iterations=1)
    # cv.imshow('Dilated Image', dialate_bird)

    # Saves to output folder
    cv.imwrite(os.path.join(output_dir, f"gray_{image_file}"), sosad_blacknwhite)
    cv.imwrite(os.path.join(output_dir, f"blur_{image_file}"), blurry_wurry)
    cv.imwrite(os.path.join(output_dir, f"edges_{image_file}"), edges) 
    cv.imwrite(os.path.join(output_dir, f"dilate_{image_file}"), dialate_bird)

    # cv.waitKey(1)  

# cv.waitKey(0)  
cv.destroyAllWindows()   # This line is safe to keep even in a headless environment