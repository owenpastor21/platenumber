import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import io

# Load the specialized YOLOv8 model for license plate detection
@st.cache_resource
def load_yolo_model():
    return YOLO('models/license_plate_detector.pt') 

# Initialize EasyOCR reader
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'])

model = load_yolo_model()
ocr = load_ocr_model()

def recognize_plate(img):
    """
    Recognize license plate from an image using YOLOv8 and EasyOCR.
    """
    # Run YOLOv8 inference
    results = model(img)

    plate_number = "No plate detected"
    confidence = 0.0
    
    # Process results
    for r in results:
        boxes = r.boxes
        if boxes:
            # Get the box with highest confidence
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            box = boxes[best_idx]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Only process if confidence is above threshold
            if confidence > 0.3:
                # Crop the image to the detected license plate
                cropped_img = img[y1:y2, x1:x2]
                
                # Use EasyOCR on the cropped image
                ocr_result = ocr.readtext(cropped_img)
                
                if ocr_result:
                    # Extract text with highest confidence
                    texts = []
                    for detection in ocr_result:
                        text = detection[1]
                        ocr_conf = detection[2]
                        if ocr_conf > 0.5:  # Filter low confidence OCR results
                            texts.append(text)
                    
                    if texts:
                        plate_number = " ".join(texts).upper()
                        # Clean up the text (remove special characters, keep alphanumeric)
                        plate_number = ''.join(c for c in plate_number if c.isalnum() or c.isspace()).strip()
    
    return plate_number, confidence

st.set_page_config(page_title="Plate Number Recognition", layout="centered")

st.title("🚗 License Plate Recognition App")

st.markdown("""
**Enhanced with YOLOv8 + EasyOCR for better accuracy!**

- Click **'Take a picture'** to capture an image
- Grant camera permissions if prompted  
- The app will detect and read the license plate automatically
""")

# Camera input
camera_input = st.camera_input("Take a picture of a license plate")

if camera_input is not None:
    # Convert the image to OpenCV format
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    with st.spinner("🔍 Analyzing image for license plates..."):
        plate_text, detection_confidence = recognize_plate(img_bgr)
    
    # Display the captured image
    st.image(image, caption="Captured Image", use_container_width=True)
    
    if plate_text != "No plate detected":
        st.success(f"✅ **License Plate Detected:** `{plate_text}`")
        st.info(f"🎯 **Detection Confidence:** {detection_confidence:.2%}")
        
        # Display in a prominent way
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6; 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            border: 2px solid #4CAF50;
            margin: 20px 0;
        ">
            <h2 style="color: #4CAF50; margin: 0;">🏁 {plate_text}</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No license plate detected. Try:")
        st.markdown("""
        - Ensure the plate is clearly visible
        - Good lighting conditions
        - Camera is stable and focused
        - Plate is not too far or too close
        """)

# Footer
st.markdown("---")
st.markdown("*Powered by YOLOv8 for detection and EasyOCR for text recognition*") 
