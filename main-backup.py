import os
import cv2
from deepface import DeepFace

img1_path = "virat-1.jpg"  # Image to compare
img2_path = "virat-4.jpg"  # Image for comparison

# Image display settings
show_image1 = True  
show_image2 = True 
fixed_width = 800 

# Check if the paths exist
if not os.path.exists(img1_path):
    raise FileNotFoundError(f"Image not found at {img1_path}")
if not os.path.exists(img2_path):
    raise FileNotFoundError(f"Image not found at {img2_path}")

try:
    # Load the images using OpenCV
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise ValueError(f"Failed to load the image at {img1_path}. The file may be corrupted or in an unsupported format.")
    if img2 is None:
        raise ValueError(f"Failed to load the image at {img2_path}. The file may be corrupted or in an unsupported format.")

    # Convert the images to RGB (DeepFace expects RGB images)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Analyze the images to detect faces and get bounding boxes
    analysis1 = DeepFace.analyze(img1_rgb, actions=['age'], detector_backend='opencv')
    analysis2 = DeepFace.analyze(img2_rgb, actions=['age'], detector_backend='opencv')

    # Extract bounding box information
    bbox1 = analysis1[0]['region']  
    bbox2 = analysis2[0]['region'] 

    # Verify if the detected face matches the comparison image
    result = DeepFace.verify(img1_rgb, img2_rgb, model_name="Facenet", enforce_detection=False)
    print(f"Verification result: {result}")

    match_text = "Faces match!" if result["verified"] else "Faces do not match."
    
    # Draw match text on the images
    cv2.putText(img1, match_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result["verified"] else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img2, match_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result["verified"] else (0, 0, 255), 2, cv2.LINE_AA)

    # Draw bounding boxes around the detected faces
    cv2.rectangle(img1, (bbox1['x'], bbox1['y']), (bbox1['x'] + bbox1['w'], bbox1['y'] + bbox1['h']), (0, 255, 0), 2)
    cv2.rectangle(img2, (bbox2['x'], bbox2['y']), (bbox2['x'] + bbox2['w'], bbox2['y'] + bbox2['h']), (0, 255, 0), 2)

    # Resize images to fixed width while maintaining aspect ratio
    def resize_image(image, width):
        height = int(image.shape[0] * width / image.shape[1])
        return cv2.resize(image, (width, height))

    img1_resized = resize_image(img1, fixed_width) if show_image1 else None
    img2_resized = resize_image(img2, fixed_width) if show_image2 else None

    # Display the images with OpenCV if enabled
    if show_image1 and img1_resized is not None:
        cv2.imshow("Image 1", img1_resized)
    if show_image2 and img2_resized is not None:
        cv2.imshow("Image 2", img2_resized)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
