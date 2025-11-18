"""
Camera Face Detection Loop
Continuously monitors camera, detects faces, and sends images to Hugging Face API
Uses hardware-specific camera initialization for PrimeX autocar (SSIG-main method)
"""

import cv2
import time
import requests
import base64
import os
import tempfile
from datetime import datetime
from typing import Optional

# Try to import pop.Util for hardware-specific camera (PrimeX autocar)
try:
    from pop import Util
    POP_UTIL_AVAILABLE = True
except ImportError:
    POP_UTIL_AVAILABLE = False
    print("Note: pop.Util not available. Will try standard camera initialization.")
    print("For PrimeX autocar hardware, ensure pop module is installed.")

# Try to import gradio_client (recommended method)
try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    print("Note: gradio_client not available. Install with: pip install gradio_client")
    print("Falling back to HTTP API method.")

class CameraFaceDetector:
    def __init__(self, huggingface_api_url: str = None, huggingface_api_token: Optional[str] = None, use_gradio_client: bool = True, width: int = 640, height: int = 480):
        """
        Initialize the camera face detector
        
        Args:
            huggingface_api_url: URL of the Hugging Face Space (optional if using gradio_client)
            huggingface_api_token: Optional API token for Hugging Face
            use_gradio_client: Use gradio_client library (recommended) or HTTP API
            width: Camera frame width (default: 640)
            height: Camera frame height (default: 480)
        """
        self.huggingface_api_url = huggingface_api_url
        self.huggingface_api_token = huggingface_api_token
        self.use_gradio_client = use_gradio_client and GRADIO_CLIENT_AVAILABLE
        self.gradio_client = None
        self.cap = None
        self.face_cascade = None
        self.last_face_detection_time = None
        self.face_detected = False
        self.wait_interval = 5  # Wait 5 seconds after face detection
        self.width = width
        self.height = height
        self.use_hardware_camera = POP_UTIL_AVAILABLE
        
        # Initialize gradio_client if available
        if self.use_gradio_client:
            try:
                # Use direct URL (more reliable than space name)
                space_url = "https://mayarelshamy-ppe-detection-system.hf.space"
                self.gradio_client = Client(space_url, hf_token=huggingface_api_token)
                print("Using gradio_client for API calls (recommended)")
            except Exception as e:
                print(f"Failed to initialize gradio_client: {e}")
                print("Falling back to HTTP API method")
                self.use_gradio_client = False
        
    def initialize(self):
        """Initialize camera and face detection cascade using SSIG-main method"""
        # Initialize camera using hardware-specific method (PrimeX autocar) or fallback
        if self.use_hardware_camera:
            try:
                # Use hardware-specific camera initialization (SSIG-main method)
                Util.enable_imshow()
                cam = Util.gstrmer(width=self.width, height=self.height)
                self.cap = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
                print("Using hardware-specific camera initialization (PrimeX autocar)")
            except Exception as e:
                print(f"Hardware camera initialization failed: {e}")
                print("Falling back to standard camera initialization")
                self.cap = cv2.VideoCapture(0)
                self.use_hardware_camera = False
        else:
            # Standard camera initialization (fallback)
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Get actual camera dimensions
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera initialized: {actual_width}x{actual_height}")
        
        # Load face detection cascade - try hardware path first, then fallback
        if self.use_hardware_camera:
            # Try hardware-specific path first (PrimeX autocar)
            haar_face = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(haar_face)
            if self.face_cascade.empty():
                # Fallback to standard path
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Standard path
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face detection cascade")
        
        print("Face detection cascade loaded successfully")
    
    def detect_face(self, frame):
        """Detect faces in the frame using SSIG-main detection parameters"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use SSIG-main face detection parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,  # SSIG-main parameter
            minNeighbors=5,   # SSIG-main parameter (increased from 1 for reliability)
            minSize=(100, 100)  # SSIG-main parameter
        )
        return len(faces) > 0, faces
    
    def capture_image(self, frame):
        """Capture and encode image for API"""
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return image_base64
    
    def send_to_huggingface_gradio_client(self, frame) -> dict:
        """Send image using gradio_client (recommended method)"""
        try:
            # Save frame to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            _, buffer = cv2.imencode('.jpg', frame)
            temp_file.write(buffer.tobytes())
            temp_file.close()
            
            # Use gradio_client
            result = self.gradio_client.predict(
                image=handle_file(temp_file.name),
                api_name="/predict"
            )
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return {
                "status": "success",
                "raw_response": result,
                "full_response": {"data": [result]}
            }
        except Exception as e:
            print(f"Error with gradio_client: {e}")
            return None
    
    def send_to_huggingface(self, image_base64: str) -> dict:
        """
        Send image to Hugging Face Gradio API
        Based on API documentation: /api/predict endpoint
        Format: {"data": [{"image": {"url": "data:image/jpeg;base64,..."}}]}
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.huggingface_api_token:
            headers['Authorization'] = f'Bearer {self.huggingface_api_token}'
        
        # Gradio API format based on documentation
        # Image should be provided as a dict with url (base64 data URI)
        image_data_uri = f"data:image/jpeg;base64,{image_base64}"
        
        # Format based on Gradio API: image component expects dict with url
        image_dict = {
            "url": image_data_uri
        }
        
        # Gradio API expects {"data": [image_dict]}
        payload = {
            "data": [image_dict]
        }
        
        try:
            response = requests.post(
                self.huggingface_api_url,
                json=payload,
                headers=headers,
                timeout=60  # Longer timeout for model loading
            )
            response.raise_for_status()
            result = response.json()
            
            # Gradio API returns {"data": [result_string]}
            # Extract the result from the data array
            if "data" in result and len(result["data"]) > 0:
                return {
                    "status": "success",
                    "raw_response": result["data"][0],
                    "full_response": result
                }
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error sending to Hugging Face: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response status: {e.response.status_code}")
                    print(f"Response text: {e.response.text}")
            return None
    
    def run(self):
        """Main detection loop"""
        self.initialize()
        
        print("Starting face detection loop...")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Detect face
                face_found, faces = self.detect_face(frame)
                
                # Draw rectangle around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display status
                status_text = "Face Detected - Waiting..." if self.face_detected else "Scanning..."
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('PPE Detection Camera', frame)
                
                # Handle face detection timing
                if face_found:
                    if not self.face_detected:
                        # First detection
                        self.face_detected = True
                        self.last_face_detection_time = time.time()
                        print(f"[{datetime.now()}] Face detected, waiting {self.wait_interval} seconds...")
                    else:
                        # Check if wait period has passed
                        elapsed = time.time() - self.last_face_detection_time
                        if elapsed >= self.wait_interval:
                            print(f"[{datetime.now()}] Capturing image and sending to Hugging Face...")
                            
                            # Capture and send
                            if self.use_gradio_client and self.gradio_client:
                                result = self.send_to_huggingface_gradio_client(frame)
                            else:
                                image_base64 = self.capture_image(frame)
                                result = self.send_to_huggingface(image_base64)
                            
                            if result:
                                print(f"[{datetime.now()}] Detection result: {result}")
                            else:
                                print(f"[{datetime.now()}] Failed to get detection result")
                            
                            # Reset detection state
                            self.face_detected = False
                            self.last_face_detection_time = None
                            
                            # Wait a bit before next detection
                            time.sleep(2)
                else:
                    # No face detected, reset state
                    if self.face_detected:
                        self.face_detected = False
                        self.last_face_detection_time = None
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping camera detection...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    # Hugging Face API Configuration
    # Method 1: Use gradio_client (recommended - set use_gradio_client=True)
    # Method 2: Use HTTP API (set use_gradio_client=False)
    
    HUGGINGFACE_API_URL = "https://mayarelshamy-ppe-detection-system.hf.space/api/predict"
    HUGGINGFACE_API_TOKEN = ""  # Optional: Add your token here if needed
    USE_GRADIO_CLIENT = True  # Set to False to use HTTP API instead
    
    # If using HTTP API, ensure URL is correct
    if not USE_GRADIO_CLIENT and HUGGINGFACE_API_URL and not HUGGINGFACE_API_URL.endswith('/api/predict'):
        if HUGGINGFACE_API_URL.endswith('/'):
            HUGGINGFACE_API_URL = HUGGINGFACE_API_URL + 'api/predict'
        else:
            HUGGINGFACE_API_URL = HUGGINGFACE_API_URL + '/api/predict'
        print(f"Using HTTP API: {HUGGINGFACE_API_URL}")
    elif USE_GRADIO_CLIENT:
        print("Using gradio_client (recommended method)")
    
    detector = CameraFaceDetector(
        HUGGINGFACE_API_URL if not USE_GRADIO_CLIENT else None,
        HUGGINGFACE_API_TOKEN if HUGGINGFACE_API_TOKEN else None,
        use_gradio_client=USE_GRADIO_CLIENT
    )
    detector.run()

