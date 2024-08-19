# Face Comparison using DeepFace and OpenCV

## Requirements
Before running the script, ensure that you have the following Python packages installed:

- opencv-python
- deepface
### You can install them using pip:
```
pip install opencv-python deepface
```
## Usage
- Images to Compare:
```
img1_path = "images/virat-1.jpg"  # Image to compare
img2_path = "images/virat-4.jpg"  # Image for comparison

```
- Image Display Settings:
```
show_image1 = True
show_image2 = True 
```

## Note:
- The script assumes that the input images contain only one face. If there are multiple faces, the first detected face is used for verification.
- The script uses the Facenet model for face verification, but this can be changed by modifying the model_name parameter in the verify function.
- If you experience issues with image loading, ensure the images are in a supported format and not corrupted.
