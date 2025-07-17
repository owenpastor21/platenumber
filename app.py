import streamlit as st
import cv2
import numpy as np
import base64
import io
from openai import OpenAI
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv12 small model for general object detection
@st.cache_resource
def load_yolo_model():
    return YOLO('models/yolo12s.pt')  # YOLOv12 small model

model = load_yolo_model()

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    try:
        api_key = st.secrets["openai"]["api_key"]
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"OpenAI API key not found. Please add it to .streamlit/secrets.toml")
        return None

def encode_image_to_base64(image):
    """
    Convert an image to base64 string for OpenAI API.
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_plate_with_gpt4o(image, client):
    """
    Use GPT-4o vision to analyze the license plate image and extract text.
    """
    if client is None:
        return "API Error"
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image)
        
        print("\n" + "="*50)
        print("🤖 MAKING GPT-4O-MINI API CALL")
        print("="*50)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a license plate recognition expert. Analyze this image and extract ONLY the license plate number/text.

Rules:
- Return ONLY the alphanumeric characters you see on the license plate
- Remove any spaces, dashes, or special characters
- If you see multiple text elements, return only the main license plate number
- If no clear license plate text is visible, return "UNCLEAR"
- Be very accurate - this is for official recognition

License plate text:"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature=0
        )
        
        # Print token usage information
        usage = response.usage
        print(f"📊 TOKEN USAGE:")
        print(f"   • Input tokens (prompt + image): {usage.prompt_tokens}")
        print(f"   • Output tokens (response): {usage.completion_tokens}")
        print(f"   • Total tokens: {usage.total_tokens}")
        
        # Calculate approximate cost (as of 2024 pricing)
        # GPT-4o-mini pricing: $0.15/1M input tokens, $0.60/1M output tokens
        input_cost = (usage.prompt_tokens / 1_000_000) * 0.15
        output_cost = (usage.completion_tokens / 1_000_000) * 0.60
        total_cost = input_cost + output_cost
        
        print(f"💰 ESTIMATED COST:")
        print(f"   • Input cost: ${input_cost:.6f}")
        print(f"   • Output cost: ${output_cost:.6f}")
        print(f"   • Total cost: ${total_cost:.6f}")
        
        result = response.choices[0].message.content.strip().upper()
        print(f"🎯 GPT-4O-MINI RESPONSE: '{result}'")
        print("="*50)
        
        # Clean the result
        result = ''.join(c for c in result if c.isalnum())
        
        return result if result and result != "UNCLEAR" else "Could not read plate"
        
    except Exception as e:
        print(f"❌ GPT-4O ERROR: {str(e)}")
        print("="*50)
        st.error(f"GPT-4o Error: {str(e)}")
        return "API Error"

def recognize_plate(img):
    """
    Recognize license plate from an image using YOLOv12 and GPT-4o vision.
    """
    client = get_openai_client()
    
    # Run YOLOv12 inference to detect all objects
    results = model(img)

    plate_number = "No plate detected"
    confidence = 0.0
    cropped_img = None
    use_full_image = False
    
    # Process results to find car-related objects that might contain license plates
    for r in results:
        boxes = r.boxes
        if boxes:
            # Look for cars, trucks, buses, motorcycles, or any vehicle
            # COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
            vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            
            vehicle_found = False
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])
                
                # If we detect a vehicle with confidence higher than 50%
                if class_id in vehicle_classes and confidence_score > 0.5:
                    vehicle_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = confidence_score
                    
                    # Crop the vehicle area (license plate likely to be in this region)
                    cropped_img = img[y1:y2, x1:x2]
                    
                    # Add some padding to the cropped image for better analysis
                    height, width = cropped_img.shape[:2]
                    pad_height = int(height * 0.1)
                    pad_width = int(width * 0.1)
                    
                    # Create padded image
                    padded_img = cv2.copyMakeBorder(
                        cropped_img, 
                        pad_height, pad_height, pad_width, pad_width,
                        cv2.BORDER_CONSTANT, 
                        value=[255, 255, 255]  # White padding
                    )
                    
                    # Use GPT-4o to analyze the cropped vehicle image for license plates
                    plate_text = analyze_plate_with_gpt4o(padded_img, client)
                    
                    if plate_text and plate_text not in ["Could not read plate", "API Error"]:
                        plate_number = plate_text
                        break  # Found a readable plate, stop looking
            
            # If no vehicle detected with confidence > 50%, analyze the full image
            if not vehicle_found:
                print(f"🔍 No vehicle detected with confidence > 50%. Analyzing full image with GPT-4o...")
                use_full_image = True
                # Use GPT-4o to analyze the full image
                plate_text = analyze_plate_with_gpt4o(img, client)
                if plate_text and plate_text not in ["Could not read plate", "API Error"]:
                    plate_number = plate_text
                    confidence = 0.3  # Lower confidence for full image analysis
    
    # If no boxes detected at all, analyze the full image
    if not results[0].boxes:
        print(f"🔍 No objects detected by YOLO. Analyzing full image with GPT-4o...")
        use_full_image = True
        plate_text = analyze_plate_with_gpt4o(img, client)
        if plate_text and plate_text not in ["Could not read plate", "API Error"]:
            plate_number = plate_text
            confidence = 0.3  # Lower confidence for full image analysis
    
    return plate_number, confidence, cropped_img, use_full_image

st.set_page_config(page_title="BIDA-Plate Recognizer", layout="centered")

st.title("🚗 BIDA-Plate Recognizer")

st.markdown("""
**AI-Powered License Plate Recognition System**

- Click **'Take a picture'** to capture an image
- Grant camera permissions if prompted  
- Advanced AI will analyze and read the license plate text
""")

# Check if API key is configured
client = get_openai_client()
if client is None:
    st.error("⚠️ **Setup Required**: Please add your OpenAI API key to `.streamlit/secrets.toml`")
    st.info("Get your API key from: https://platform.openai.com/api-keys")
    st.stop()

# Camera input
camera_input = st.camera_input("📷 Capture License Plate Image")

if camera_input is not None:
    # Convert the image to OpenCV format
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    with st.spinner("🤖 AI is analyzing the image..."):
        plate_text, detection_confidence, cropped_vehicle, use_full_image = recognize_plate(img_bgr)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(image, caption="Captured Image", use_container_width=True)
    
    with col2:
        if use_full_image:
            st.subheader("🔍 Analysis Method")
            st.info("**Full Image Analysis**\n\nNo vehicle detected with confidence > 50%. GPT-4o analyzed the entire image for license plates.")
        elif cropped_vehicle is not None:
            st.subheader("🚗 Detected Vehicle")
            # Convert BGR to RGB for display
            cropped_rgb = cv2.cvtColor(cropped_vehicle, cv2.COLOR_BGR2RGB)
            st.image(cropped_rgb, caption="Vehicle Containing License Plate", use_container_width=True)
    
    # Results section
    st.markdown("---")
    
    if plate_text != "No plate detected":
        st.success(f"✅ **License Plate Detected**")
        
        # Show different confidence information based on analysis method
        if use_full_image:
            st.info(f"🔍 **Analysis Method:** Full Image (GPT-4o Vision)")
            st.info(f"📊 **Detection Status:** Vehicle confidence < 50%, used AI vision")
        else:
            st.info(f"🎯 **Vehicle Detection Confidence:** {detection_confidence:.1%}")
            st.info(f"🔍 **Analysis Method:** Vehicle-focused (YOLO + GPT-4o)")
        
        # Display the plate number prominently
        st.markdown(f"""
        <div style="
            background-color: #e8f5e8; 
            padding: 25px; 
            border-radius: 15px; 
            text-align: center;
            border: 3px solid #28a745;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="color: #155724; margin: 0; font-family: 'Courier New', monospace; letter-spacing: 3px;">
                {plate_text}
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.warning("⚠️ No license plate detected")
        
        if use_full_image:
            st.info("🔍 **Analysis Performed**: Full image analyzed with GPT-4o Vision")
        else:
            st.info("🚗 **Analysis Performed**: Vehicle-focused detection")
            
        st.markdown("""
        **Tips for better detection:**
        - Ensure the plate is clearly visible and well-lit
        - Avoid shadows and reflections
        - Keep the camera steady and focused
        - Position the plate within the center of the frame
        """)

# Footer
st.markdown("---")
st.markdown("*BIDA-Plate Recognizer*")
