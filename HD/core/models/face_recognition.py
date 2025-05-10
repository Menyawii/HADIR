import os
import cv2
import numpy as np
import time
import logging
from typing import Dict, Optional, List
from deepface import DeepFace

from core.utils.config import Config
from core.utils.exceptions import SystemInitializationError, FaceRecognitionError
from core.utils.encoding_cache import EncodingCache
from core.models.image_processor import ImagePreprocessor
from core.models.liveness_detection import LivenessDetector
from data.structures import RecognitionResult

class FaceRecognitionSystem:
    """Handles student face registration & verification with lightweight liveness."""

    def __init__(self):
        try:
            self.encoding_cache = EncodingCache()
            self.image_preprocessor = ImagePreprocessor()
            self.liveness_detector = LivenessDetector()
            self.stored_images = self._load_stored_images()
            self._cache_stored_images()

            # Performance monitoring
            self.metrics = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "avg_time": 0.0
            }
        except Exception as e:
            logging.error(f"Init failed: {e}")
            raise SystemInitializationError(str(e))

    def _load_stored_images(self) -> List[str]:
        """Load stored face images"""
        os.makedirs(Config.STORED_IMAGES_DIR, exist_ok=True)
        return [
            os.path.join(Config.STORED_IMAGES_DIR, f)
            for f in os.listdir(Config.STORED_IMAGES_DIR)
            if f.lower().endswith(".jpg")
        ]


    def _cache_stored_images(self):
        """Pre-cache encodings for stored images"""
        for path in self.stored_images:
            self.encoding_cache.get_encoding(path)

    def get_face_encoding_for_storage(self, img: np.ndarray) -> Dict:
        """Generate face encoding for registration"""
        try:
            # Add debug logging
            print(f"Input image shape: {img.shape}")
            print(f"Input image dtype: {img.dtype}")
            
            preprocessed = self.image_preprocessor.preprocess_image(img)
            if preprocessed is None:
                print("Preprocessing returned None")
                return {
                    "success": False,
                    "message": "Preprocessing failed",
                    "encoding": None
                }

            # Add debug logging for preprocessed image
            print(f"Preprocessed image shape: {preprocessed.shape}")
            print(f"Preprocessed image dtype: {preprocessed.dtype}")

            # Save preprocessed image temporarily
            temp_path = f"temp_preprocessed_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))
            print(f"Saved temporary image to: {temp_path}")

            try:
                encoding = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=True  # Add this parameter
                )
                
                if encoding:
                    emb = encoding[0]["embedding"]
                    if isinstance(emb, np.ndarray):
                        emb = emb.tolist()
                    return {
                        "success": True,
                        "encoding": emb,
                        "message": "OK"
                    }
                
            except Exception as deep_face_error:
                print(f"DeepFace error: {str(deep_face_error)}")
                return {
                    "success": False,
                    "message": f"DeepFace processing failed: {str(deep_face_error)}",
                    "encoding": None
                }
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            return {
                "success": False,
                "message": "Encoding failed",
                "encoding": None
            }

        except Exception as e:
            print(f"General error in face encoding: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "encoding": None
            }

    def verify_student(self, student_id: str, captured_image: np.ndarray) -> RecognitionResult:
        """
        Verify a student’s identity by comparing the captured face to the stored
        face encoding.  Liveness detection is DISABLED / BYPASSED in this version.
        """
        start_time = time.time()
        self.metrics["attempts"] += 1

        try:
            # -----------------------------------------------------------
            # 0)  Basic logging / sanity-checks
            # -----------------------------------------------------------
            print(f"[Verify] Student ID  : {student_id}")
            print(f"[Verify] Image shape: {captured_image.shape}")

            # -----------------------------------------------------------
            # 1)  (Disabled) Liveness check  →  force success
            # -----------------------------------------------------------
            live_result = {"live": True,
                           "score": 1.0,
                           "message": "Liveness check bypassed"}
            print(f"[Verify] Liveness bypass: {live_result}")

            # -----------------------------------------------------------
            # 2)  Retrieve stored face encoding for this student
            # -----------------------------------------------------------
            stored_repr = self.get_student_encoding(student_id)
            if stored_repr is None:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="No stored profile found",
                    verification_type="storage"
                )

            # -----------------------------------------------------------
            # 3)  Generate encoding for the captured image
            # -----------------------------------------------------------
            preprocessed = self.image_preprocessor.preprocess_image(captured_image)
            if preprocessed is None:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="Failed to preprocess image",
                    verification_type="preprocessing"
                )

            temp_path = f"temp_verify_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))

            try:
                live_repr = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=True
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if not live_repr:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="Failed to generate encoding",
                    verification_type="encoding"
                )

            # -----------------------------------------------------------
            # 4)  Compare encodings (cosine similarity)
            # -----------------------------------------------------------
            a = np.array(live_repr[0]["embedding"])
            b = np.array(stored_repr[0]["embedding"])
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            distance = 1.0 - similarity
            verified = distance <= Config.FACE_RECOGNITION_THRESHOLD

            # -----------------------------------------------------------
            # 5)  Update metrics
            # -----------------------------------------------------------
            elapsed = time.time() - start_time
            if verified:
                self.metrics["successes"] += 1
                prev = self.metrics["successes"] - 1
                self.metrics["avg_time"] = ((self.metrics["avg_time"] * prev) + elapsed) / self.metrics["successes"]
            else:
                self.metrics["failures"] += 1

            # -----------------------------------------------------------
            # 6)  Build and return result object
            # -----------------------------------------------------------
            return RecognitionResult(
                success=verified,
                confidence_score=similarity,
                verification_time=elapsed,
                verification_type="face",
                data={"distance": distance}
            )

        except Exception as e:
            # Generic catch-all for unexpected errors
            logging.error(f"Verification error: {e}")
            self.metrics["failures"] += 1
            return RecognitionResult(
                success=False,
                error_message=str(e),
                verification_type="error"
            )

    def get_student_encoding(self, student_id: str) -> Optional[List]:
        """Get stored face encoding for a student"""
        path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
        return self.encoding_cache.get_encoding(path) if os.path.exists(path) else None

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {**self.metrics, "cached": len(self.stored_images)}
    def verify_student_images(self, stored_image: np.ndarray, 
                            captured_image: np.ndarray) -> Dict:
        """Compare two face images directly and return similarity"""
        try:
            # Preprocess both images
            stored_processed = self.image_preprocessor.preprocess_image(stored_image)
            captured_processed = self.image_preprocessor.preprocess_image(captured_image)
            
            if stored_processed is None or captured_processed is None:
                return {
                    "success": False,
                    "confidence_score": 0.0,
                    "message": "Failed to preprocess one or both images"
                }
                
            # Save temporary files
            temp_stored = f"temp_stored_{int(time.time())}.jpg"
            temp_captured = f"temp_captured_{int(time.time())}.jpg"
            
            cv2.imwrite(temp_stored, (stored_processed * 255).astype(np.uint8))
            cv2.imwrite(temp_captured, (captured_processed * 255).astype(np.uint8))
            
            try:
                # Get representations
                stored_repr = DeepFace.represent(
                    img_path=temp_stored,
                    model_name="Facenet",
                    enforce_detection=True
                )
                
                captured_repr = DeepFace.represent(
                    img_path=temp_captured,
                    model_name="Facenet",
                    enforce_detection=True
                )
                
                # Compare representations
                if stored_repr and captured_repr:
                    a = np.array(stored_repr[0]["embedding"])
                    b = np.array(captured_repr[0]["embedding"])
                    
                    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                    distance = 1.0 - similarity
                    
                    return {
                        "success": distance <= Config.FACE_RECOGNITION_THRESHOLD,
                        "confidence_score": similarity,
                        "distance": distance
                    }
                else:
                    return {
                        "success": False,
                        "confidence_score": 0.0,
                        "message": "Failed to generate representations"
                    }
                    
            finally:
                # Clean up temporary files
                for temp_file in [temp_stored, temp_captured]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
        except Exception as e:
            logging.error(f"Error comparing images: {str(e)}")
            return {
                "success": False,
                "confidence_score": 0.0,
                "message": str(e)
            }