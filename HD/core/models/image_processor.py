import cv2
import numpy as np
from typing import Optional, Tuple
from core.utils.config import Config
import logging

class ImagePreprocessor:
    """Handles image preprocessing for face recognition"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = Config.IMAGE_SIZE) -> np.ndarray:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)

    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.3, beta: int = 5) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def detect_and_align_face(image: np.ndarray) -> Optional[np.ndarray]:
        """Detects and aligns face in image"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = []
            scale_factors = [1.1, 1.2, 1.3]
            min_neighbors_options = [3, 4, 5]
            
            for scale in scale_factors:
                for min_neighbors in min_neighbors_options:
                    detected = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(50, 50),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(detected) > 0:
                        faces = detected
                        break
                if len(faces) > 0:
                    break
            
            if len(faces) == 0:
                return None
            
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            
            padding = 30
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            face_roi = image[y:y+h, x:x+w]
            
            if not ImagePreprocessor.check_face_quality(face_roi):
                return None
            
            return ImagePreprocessor.resize_image(face_roi)
            
        except Exception as e:
            logging.error(f"Face detection error: {str(e)}")
            return None

    @staticmethod
    def check_face_quality(face_image: np.ndarray) -> bool:
        """Basic quality check"""
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return False
            
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            brightness = np.mean(gray)
            if brightness < 20 or brightness > 250:
                return False
            
            contrast = np.std(gray)
            if contrast < 10:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Face quality check error: {str(e)}")
            return False

    @staticmethod
    def preprocess_image(image: np.ndarray) -> Optional[np.ndarray]:
        """Complete preprocessing pipeline"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            enhanced = ImagePreprocessor.adjust_brightness_contrast(image)
            face_img = ImagePreprocessor.detect_and_align_face(enhanced)
            
            if face_img is None:
                return None

            face_img = ImagePreprocessor.normalize_image(face_img)
            return face_img

        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            return None