from deepface import DeepFace 
import cv2
import os
import time
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='facial_recognition.log'
)

class FaceRecognitionError(Exception):
    """Custom exception for face recognition related errors"""
    pass

class ImagePreprocessor:
    """Class for handling image preprocessing operations"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to range [0,1]"""
        return image.astype('float32') / 255.0

    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, 
                                 alpha: float = 1.0, 
                                 beta: int = 0) -> np.ndarray:
        """Adjust image brightness and contrast"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def detect_and_align_face(image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face in image and align it"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                logging.warning("No face detected in image")
                return None
            
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = image[y:y+h, x:x+w]
            face_roi = ImagePreprocessor.resize_image(face_roi)
            
            return face_roi
            
        except Exception as e:
            logging.error(f"Error in face detection and alignment: {str(e)}")
            return None

    @staticmethod
    def preprocess_image(image: np.ndarray) -> Optional[np.ndarray]:
        """Apply full preprocessing pipeline to image"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            face_img = ImagePreprocessor.detect_and_align_face(image)
            if face_img is None:
                return None

            face_img = ImagePreprocessor.adjust_brightness_contrast(
                face_img, 
                alpha=1.5,
                beta=10
            )

            face_img = ImagePreprocessor.normalize_image(face_img)

            logging.info("Successfully preprocessed image")
            return face_img

        except Exception as e:
            logging.error(f"Error in image preprocessing: {str(e)}")
            return None

def capture_image() -> Optional[str]:
    """Capture and preprocess image from webcam"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise FaceRecognitionError("Failed to open webcam")

        ret, frame = cap.read()
        if not ret:
            raise FaceRecognitionError("Failed to capture frame from webcam")

        preprocessed_frame = ImagePreprocessor.preprocess_image(frame)
        if preprocessed_frame is None:
            raise FaceRecognitionError("Failed to preprocess captured image")

        filename = f"captured_image_{int(time.time())}.jpg"
        cv2.imwrite(filename, (preprocessed_frame * 255).astype(np.uint8))
        logging.info(f"Successfully captured and preprocessed image: {filename}")
        return filename

    except Exception as e:
        logging.error(f"Error in capture_image: {str(e)}")
        return None

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

def generate_encoding(image_path: str) -> Optional[List]:
    """Generate face encoding from preprocessed image"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise FaceRecognitionError(f"Cannot read image file: {image_path}")

        preprocessed_img = ImagePreprocessor.preprocess_image(img)
        if preprocessed_img is None:
            raise FaceRecognitionError("Failed to preprocess image")

        temp_path = f"temp_preprocessed_{os.path.basename(image_path)}"
        cv2.imwrite(temp_path, (preprocessed_img * 255).astype(np.uint8))

        encoding = DeepFace.represent(img_path=temp_path, model_name="Facenet")
        
        os.remove(temp_path)
        
        logging.info(f"Successfully generated encoding for {image_path}")
        return encoding

    except Exception as e:
        logging.error(f"Error generating encoding for {image_path}: {str(e)}")
        return None

def verify_face(new_encoding: Optional[List]) -> str:
    """Verify face against stored images"""
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

def cleanup_temp_files(keep_files=False):
    """Clean up temporary captured images"""
    if keep_files:
        logging.info("Keeping temporary files for debugging")
        return

    try:
        for file in os.listdir():
            if file.startswith('captured_image_') or file.startswith('temp_preprocessed_'):
                os.remove(file)
                logging.info(f"Cleaned up temporary file: {file}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary files: {str(e)}")

def main():
    """Main function with preprocessing pipeline"""
    try:
        image_filename = capture_image()
        if image_filename is None:
            raise FaceRecognitionError("Failed to capture and preprocess image")

        new_encoding = generate_encoding(image_filename)
        if new_encoding is None:
            raise FaceRecognitionError("Failed to generate encoding")

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