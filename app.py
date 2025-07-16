import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 model
model = YOLO('models/yolov8n.pt') 

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def recognize_plate(img):
    """
    Recognize license plate from an image using YOLOv8 and EasyOCR.
    """
    # Run YOLOv8 inference
    results = model(img)

    plate_number = "No plate detected"
    
    # Process results
    for r in results:
        boxes = r.boxes
        if boxes:
            # Get the first detected box
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the image to the detected license plate
            cropped_img = img[y1:y2, x1:x2]
            
            # Use EasyOCR on the cropped image
            ocr_result = reader.readtext(cropped_img)
            
            if ocr_result:
                # Assuming the first detected text is the plate number
                plate_number = ocr_result[0][1]
                return plate_number

    return None


st.set_page_config(page_title="Plate Number Recognition", layout="centered")

st.title("License Plate Recognition App")

st.markdown("""
This app uses your phone's camera to recognize license plates.
- Click the **'Take a picture'** button below.
- Grant camera permissions if prompted.
- Capture a clear image of the license plate.
""")

picture = st.camera_input("Take a picture")

if picture:
    # To read image file buffer in OpenCV
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.image(cv2_img, channels="BGR", caption="Your Picture", use_column_width=True)
    
    with st.spinner('Recognizing plate number...'):
        plate = recognize_plate(cv2_img)
        
        if plate:
            st.success(f"Detected Plate Number: **{plate}**")
        else:
            st.error("Could not recognize the license plate. Please try again with a clearer picture.") 