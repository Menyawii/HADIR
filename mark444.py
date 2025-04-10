from deepface import DeepFace 
import cv2
import os
import time
import logging
from typing import Optional, List, Dict
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='facial_recognition.log'
)

class FaceRecognitionError(Exception):
    """Custom exception for face recognition related errors"""
    pass

def capture_image() -> Optional[str]:
    """
    Capture image from webcam with error handling
    
    Returns:
        str: Path to the captured image file
        None: If capture fails
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise FaceRecognitionError("Failed to open webcam")

        ret, frame = cap.read()
        if not ret:
            raise FaceRecognitionError("Failed to capture frame from webcam")

        # Check if frame is empty or invalid
        if frame is None or frame.size == 0:
            raise FaceRecognitionError("Captured frame is empty or invalid")

        filename = f"captured_image_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        logging.info(f"Successfully captured image: {filename}")
        return filename

    except Exception as e:
        logging.error(f"Error in capture_image: {str(e)}")
        return None

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

def generate_encoding(image_path: str) -> Optional[List]:
    """
    Generate face encoding from image with error handling
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        List: Face encoding if successful
        None: If encoding fails
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Verify image can be read
        img = cv2.imread(image_path)
        if img is None:
            raise FaceRecognitionError(f"Cannot read image file: {image_path}")

        encoding = DeepFace.represent(img_path=image_path, model_name="Facenet")
        logging.info(f"Successfully generated encoding for {image_path}")
        return encoding

    except FileNotFoundError as e:
        logging.error(f"File not found error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error generating encoding for {image_path}: {str(e)}")
        return None

def verify_face(new_encoding: Optional[List]) -> str:
    """
    Verify face against stored images with error handling
    
    Args:
        new_encoding (List): Encoding of the captured face
    
    Returns:
        str: Verification result message
    """
    try:
        if new_encoding is None:
            raise FaceRecognitionError("No face encoding provided")

        matches = []
        stored_images = [f for f in os.listdir() if f.endswith('.jpg') and not f.startswith('captured_image')]

        if not stored_images:
            raise FaceRecognitionError("No stored images found for comparison")

        for filename in stored_images:
            try:
                result = DeepFace.verify(
                    img1_path=new_encoding[0]['embedding'],
                    img2_path=filename,
                    model_name="Facenet"
                )
                
                if result.get('verified', False):
                    matches.append(filename)
                    logging.info(f"Match found with {filename}")

            except Exception as e:
                logging.warning(f"Error verifying with {filename}: {str(e)}")
                continue

        if matches:
            return f"Matches found with: {', '.join(matches)}."
        return "No match found."

    except Exception as e:
        error_msg = f"Error in face verification: {str(e)}"
        logging.error(error_msg)
        return error_msg

def cleanup_temp_files():
    """Clean up temporary captured images"""
    try:
        for file in os.listdir():
            if file.startswith('captured_image_') and file.endswith('.jpg'):
                os.remove(file)
                logging.info(f"Cleaned up temporary file: {file}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary files: {str(e)}")

def main():
    """Main function with error handling"""
    try:
        # Capture a new image
        image_filename = capture_image()
        if image_filename is None:
            raise FaceRecognitionError("Failed to capture image")

        # Generate encoding for the captured image
        new_encoding = generate_encoding(image_filename)
        if new_encoding is None:
            raise FaceRecognitionError("Failed to generate encoding")

        # Verify the encoding against stored images
        result = verify_face(new_encoding)
        print(result)
        logging.info(f"Verification result: {result}")

    except Exception as e:
        error_msg = f"Error in main execution: {str(e)}"
        logging.error(error_msg)
        print(error_msg)

    finally:
        cleanup_temp_files()

if __name__ == "__main__":
    main()