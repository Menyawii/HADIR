# ===============================
# IMPORTS AND BASIC SETUP
# ===============================
from deepface import DeepFace 
import cv2
import os
import sys
import time
import logging
import numpy as np
import pickle
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple

# ===============================
# LOGGING CONFIGURATION
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='facial_recognition.log'
)

# ===============================
# CUSTOM EXCEPTIONS
# ===============================
class FaceRecognitionError(Exception):
    """Custom exception for face recognition related errors"""
    pass

# ===============================
# PROGRESS INDICATOR CLASS
# ===============================
class ProgressIndicator:
    """Handles visual feedback for user"""
    @staticmethod
    def show_status(message: str, end: str = '\n'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", end=end)
        sys.stdout.flush()

    @staticmethod
    def show_success(message: str):
        print(f"\n✅ {message}")

    @staticmethod
    def show_error(message: str):
        print(f"\n❌ {message}")

    @staticmethod
    def show_warning(message: str):
        print(f"\n⚠️ {message}")

# ===============================
# ENCODING CACHE CLASS
# ===============================
class EncodingCache:
    """Handles caching of face encodings"""
    def __init__(self, cache_file: str = 'encodings_cache.pkl'):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return {}

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def get_encoding(self, image_path: str) -> Optional[List]:
        if image_path in self.cache:
            logging.info(f"Retrieved encoding from cache for {image_path}")
            return self.cache[image_path]
        
        encoding = generate_encoding(image_path)
        if encoding is not None:
            self.cache[image_path] = encoding
            self.save_cache()
        return encoding

# ===============================
# IMAGE PREPROCESSING CLASS
# ===============================
class ImagePreprocessor:
    """Handles all image preprocessing operations"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        return image.astype('float32') / 255.0

    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, 
                                 alpha: float = 1.0, 
                                 beta: int = 0) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def detect_and_align_face(image: np.ndarray) -> Optional[np.ndarray]:
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

# ===============================
# CAMERA HANDLING
# ===============================
@contextmanager
def camera_context():
    """Context manager for camera handling"""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        yield cap
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

# ===============================
# IMAGE CAPTURE AND ENCODING
# ===============================
def capture_image() -> Optional[str]:
    """Capture and preprocess image from webcam"""
    try:
        ProgressIndicator.show_status("Initializing camera...")
        with camera_context() as cap:
            if not cap.isOpened():
                raise FaceRecognitionError("Failed to open webcam")

            ProgressIndicator.show_status("Camera ready. Capturing image...")
            ret, frame = cap.read()
            if not ret:
                raise FaceRecognitionError("Failed to capture frame")

            ProgressIndicator.show_status("Processing captured image...")
            preprocessed_frame = ImagePreprocessor.preprocess_image(frame)
            if preprocessed_frame is None:
                raise FaceRecognitionError("Failed to preprocess image")

            filename = f"captured_image_{int(time.time())}.jpg"
            cv2.imwrite(filename, (preprocessed_frame * 255).astype(np.uint8))
            
            ProgressIndicator.show_success(f"Image captured and saved as: {filename}")
            return filename

    except Exception as e:
        ProgressIndicator.show_error(f"Error during image capture: {str(e)}")
        logging.error(f"Error in capture_image: {str(e)}")
        return None

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

# ===============================
# FACE RECOGNITION SYSTEM
# ===============================
class FaceRecognitionSystem:
    def __init__(self):
        ProgressIndicator.show_status("Initializing Face Recognition System...")
        self.encoding_cache = EncodingCache()
        self.image_preprocessor = ImagePreprocessor()
        self.stored_images = self._load_stored_images()
        self._cache_stored_images()
        ProgressIndicator.show_success("System initialized successfully")

    def _load_stored_images(self) -> List[str]:
        ProgressIndicator.show_status("Loading stored images...")
        images = [f for f in os.listdir() 
                 if f.endswith('.jpg') and 
                 not f.startswith('captured_image_')]
        ProgressIndicator.show_status(f"Found {len(images)} stored images")
        return images

    def _cache_stored_images(self):
        """Pre-cache encodings for stored images"""
        total = len(self.stored_images)
        ProgressIndicator.show_status(f"Caching encodings for {total} stored images...")
        
        for idx, image in enumerate(self.stored_images, 1):
            ProgressIndicator.show_status(
                f"Processing image {idx}/{total}: {image}...",
                end='\r'
            )
            self.encoding_cache.get_encoding(image)
        
        print()  # New line after progress updates
        ProgressIndicator.show_success("All stored images cached successfully")

    def process_verification(self, captured_image_path: str) -> Dict:
        try:
            ProgressIndicator.show_status("Generating encoding for captured image...")
            new_encoding = self.encoding_cache.get_encoding(captured_image_path)
            if new_encoding is None:
                raise FaceRecognitionError("Failed to generate encoding")

            matches = []
            confidence_scores = {}
            total = len(self.stored_images)

            ProgressIndicator.show_status("Starting face verification process...")
            for idx, stored_image in enumerate(self.stored_images, 1):
                ProgressIndicator.show_status(
                    f"Comparing with stored image {idx}/{total}...",
                    end='\r'
                )
                
                stored_encoding = self.encoding_cache.get_encoding(stored_image)
                if stored_encoding is None:
                    continue

                result = DeepFace.verify(
                    img1_path=new_encoding[0]['embedding'],
                    img2_path=stored_encoding[0]['embedding'],
                    model_name="Facenet"
                )

                if result.get('verified', False):
                    matches.append(stored_image)
                    confidence_scores[stored_image] = result.get('distance', 0)

            print()  # New line after progress updates
            return {
                "status": "success",
                "matches": matches,
                "confidence_scores": confidence_scores
            }

        except Exception as e:
            ProgressIndicator.show_error(f"Verification error: {str(e)}")
            logging.error(f"Verification error: {str(e)}")
            return {"status": "error", "message": str(e)}

# ===============================
# CLEANUP AND UTILITIES
# ===============================
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

# ===============================
# MAIN EXECUTION
# ===============================
def main():
    """Main function with enhanced user feedback"""
    try:
        ProgressIndicator.show_status("Starting Face Recognition System...")
        system = FaceRecognitionSystem()
        
        # Capture and preprocess image
        ProgressIndicator.show_status("\nPreparing to capture image...")
        image_path = capture_image()
        if image_path is None:
            raise FaceRecognitionError("Failed to capture image")

        # Process verification
        ProgressIndicator.show_status("\nStarting verification process...")
        result = system.process_verification(image_path)
        
        if result["status"] == "success":
            if result["matches"]:
                ProgressIndicator.show_success("Matches found!")
                print("\nMatch details:")
                for match in result["matches"]:
                    confidence = result["confidence_scores"][match]
                    print(f"- {match} (confidence: {confidence:.3f})")
            else:
                ProgressIndicator.show_warning("No matches found in the database")
        else:
            ProgressIndicator.show_error(f"Error: {result['message']}")

    except Exception as e:
        error_msg = f"Error in main execution: {str(e)}"
        ProgressIndicator.show_error(error_msg)
        logging.error(error_msg)

    finally:
        ProgressIndicator.show_status("\nCleaning up temporary files...")
        cleanup_temp_files()
        ProgressIndicator.show_status("Process completed")

if __name__ == "__main__":
    main()