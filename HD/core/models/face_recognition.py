import os
import cv2
import numpy as np
import time
import logging
import json
from typing import Dict, Optional, List
import tensorflow as tf
from deepface import DeepFace
from core.utils.config import Config
from core.utils.exceptions import SystemInitializationError, FaceRecognitionError
from core.utils.encoding_cache import EncodingCache
from core.models.image_processor import ImagePreprocessor
from core.models.liveness_detection import LivenessDetector
from data.structures import RecognitionResult

class FaceRecognitionSystem:
    """Mobile-optimized face recognition system with Facenet and in-memory processing."""

    def __init__(self):
        try:
            self.encoding_cache = EncodingCache()
            self.image_preprocessor = ImagePreprocessor()
            self.liveness_detector = LivenessDetector()
            self.stored_images = self._load_stored_images()
            self._cache_stored_images()
            self.multiple_encodings = {}
            self.encoding_weights = {}
            self.student_thresholds = {}
            self._load_multiple_encodings()
            
            # Load models optimized for mobile deployment
            self._load_face_models()
            
            self.metrics = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "avg_time": 0.0
            }
            
        except Exception as e:
            logging.error(f"Init failed: {e}")
            raise SystemInitializationError(str(e))

    def _load_face_models(self):
        """Load Facenet model - optimized for mobile deployment"""
        try:
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Initialize model as None - we'll use DeepFace.represent() directly
            self.facenet_model = None
            self.model_name = "Facenet"
            
            # Test DeepFace.represent() to ensure it works
            test_img = np.random.rand(160, 160, 3).astype(np.uint8)
            temp_path = "temp_test.jpg"
            cv2.imwrite(temp_path, test_img)
            
            try:
                _ = DeepFace.represent(img_path=temp_path, model_name=self.model_name, enforce_detection=False)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
                
        except Exception as e:
            logging.error(f"Error loading Facenet model: {e}")
            raise SystemInitializationError(f"Failed to load face recognition model: {e}")

    def _create_temp_file_from_array(self, img_array: np.ndarray) -> str:
        """Create a temporary file from numpy array"""
        try:
            # Ensure image is in correct format
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            # Create unique temporary filename
            temp_path = f"temp_face_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(temp_path, img_array)
            return temp_path
            
        except Exception as e:
            logging.error(f"Error creating temporary file: {e}")
            raise

    def _cleanup_temp_file(self, temp_path: str):
        """Safely clean up temporary file"""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logging.error(f"Error cleaning up temp file {temp_path}: {e}")

    def _extract_embedding_direct(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using DeepFace.represent() with minimal file I/O"""
        temp_path = None
        try:
            # Create temporary file
            temp_path = self._create_temp_file_from_array(face_img)
            
            # Get embedding using DeepFace.represent
            result = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if result and len(result) > 0:
                embedding = np.array(result[0]["embedding"])
                
                # L2 normalize embedding
                norm = np.linalg.norm(embedding)
                if norm != 0:
                    embedding = embedding / norm
                
                return embedding
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error extracting embedding: {e}")
            return None
        finally:
            if temp_path:
                self._cleanup_temp_file(temp_path)

    def _load_stored_images(self) -> List[str]:
        """Load stored face images"""
        os.makedirs(Config.STORED_IMAGES_DIR, exist_ok=True)
        return [
            os.path.join(Config.STORED_IMAGES_DIR, f)
            for f in os.listdir(Config.STORED_IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def _cache_stored_images(self):
        """Pre-cache encodings for stored images"""
        for path in self.stored_images:
            self.encoding_cache.get_encoding(path)

    def _load_multiple_encodings(self):
        """Load multiple encodings and thresholds per student"""
        multiple_encodings_file = os.path.join(Config.STORED_IMAGES_DIR, "multiple_encodings.json")
        if os.path.exists(multiple_encodings_file):
            try:
                with open(multiple_encodings_file, 'r') as f:
                    data = json.load(f)
                    self.multiple_encodings = data.get('encodings', {})
                    self.encoding_weights = data.get('weights', {})
                    self.student_thresholds = data.get('thresholds', {})
            except Exception as e:
                logging.error(f"Error loading multiple encodings: {e}")

    def _save_multiple_encodings(self):
        """Save multiple encodings and thresholds to file"""
        try:
            data = {
                'encodings': self.multiple_encodings,
                'weights': self.encoding_weights,
                'thresholds': self.student_thresholds,
                'model_version': 'facenet_only',
                'last_updated': time.time()
            }
            multiple_encodings_file = os.path.join(Config.STORED_IMAGES_DIR, "multiple_encodings.json")
            with open(multiple_encodings_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving multiple encodings: {e}")

    def _assess_image_quality(self, img: np.ndarray) -> float:
        """Assess image quality for face recognition"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            brightness = np.mean(gray)
            contrast = gray.std()
            
            sharpness_score = min(sharpness / 500.0, 1.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            contrast_score = min(contrast / 40.0, 1.0)
            
            quality = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logging.error(f"Error assessing image quality: {e}")
            return 0.5

    def _calculate_encoding_confidence(self, embedding: np.ndarray) -> float:
        """Calculate confidence score for an embedding"""
        try:
            magnitude = np.linalg.norm(embedding)
            variance = np.var(embedding)
            confidence = min((variance * 50 + magnitude * 0.5), 1.0)
            return max(confidence, 0.1)
        except:
            return 0.5

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
            similarity = np.dot(emb1_norm, emb2_norm)
            return max(0.0, min(1.0, similarity))
        except:
            return 0.0

    def _get_dynamic_threshold(self, student_id: str) -> float:
        """Get dynamic threshold for student based on historical performance"""
        base_threshold = Config.FACE_RECOGNITION_THRESHOLD
        if student_id in self.student_thresholds:
            return self.student_thresholds[student_id]
        return base_threshold

    def _update_student_threshold(self, student_id: str, similarity: float, verified: bool):
        """Update dynamic threshold based on verification results"""
        if student_id not in self.student_thresholds:
            self.student_thresholds[student_id] = Config.FACE_RECOGNITION_THRESHOLD
        
        current_threshold = self.student_thresholds[student_id]
        
        if verified and similarity > current_threshold + 0.1:
            self.student_thresholds[student_id] = min(current_threshold + 0.02, 0.9)
        elif not verified and similarity < current_threshold:
            self.student_thresholds[student_id] = max(current_threshold - 0.01, 0.3)

    def _add_multiple_encoding(self, student_id: str, encoding: np.ndarray, quality: float):
        """Add encoding to multiple encodings storage"""
        if student_id not in self.multiple_encodings:
            self.multiple_encodings[student_id] = []
            self.encoding_weights[student_id] = []
        
        if len(self.multiple_encodings[student_id]) >= 5:
            min_idx = np.argmin(self.encoding_weights[student_id])
            self.multiple_encodings[student_id].pop(min_idx)
            self.encoding_weights[student_id].pop(min_idx)
        
        encoding_list = encoding.tolist() if isinstance(encoding, np.ndarray) else encoding
        self.multiple_encodings[student_id].append(encoding_list)
        self.encoding_weights[student_id].append(quality)
        self._save_multiple_encodings()

    def get_face_encoding_for_storage(self, img: np.ndarray, student_id: str = None) -> Dict:
        """Generate face encoding with minimal file I/O"""
        try:
            quality_score = self._assess_image_quality(img)
            
            if quality_score < 0.3:
                return {
                    "success": False,
                    "message": f"Image quality too low: {quality_score:.2f}",
                    "encoding": None,
                    "quality_score": quality_score
                }
            
            preprocessed = self.image_preprocessor.preprocess_image(img)
            if preprocessed is None:
                return {
                    "success": False,
                    "message": "Face preprocessing failed - no face detected",
                    "encoding": None
                }

            if preprocessed.dtype != np.uint8:
                if preprocessed.max() <= 1.0:
                    preprocessed = (preprocessed * 255).astype(np.uint8)
                else:
                    preprocessed = preprocessed.astype(np.uint8)

            embedding = self._extract_embedding_direct(preprocessed)
            
            if embedding is not None:
                confidence = self._calculate_encoding_confidence(embedding)
                
                if student_id:
                    self._add_multiple_encoding(student_id, embedding, quality_score)
                
                return {
                    "success": True,
                    "encoding": embedding.tolist(),
                    "message": "OK",
                    "quality_score": quality_score,
                    "model_used": "Facenet",
                    "confidence": confidence
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to generate face embedding",
                    "encoding": None
                }

        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "encoding": None
            }

    def verify_student(self, student_id: str, captured_image: np.ndarray) -> RecognitionResult:
        """Enhanced verification with minimal file I/O"""
        start_time = time.time()
        self.metrics["attempts"] += 1

        try:
            live_result = self.liveness_detector.analyze(captured_image)
            
            if not live_result.get("live", True):
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message=f"Liveness check failed: {live_result.get('message', 'Unknown')}",
                    verification_type="liveness",
                    data={"liveness_score": live_result.get("score", 0.0)}
                )

            stored_encodings = self.multiple_encodings.get(student_id, [])
            if not stored_encodings:
                stored_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
                if os.path.exists(stored_path):
                    stored_img = cv2.imread(stored_path)
                    if stored_img is not None:
                        result = self.get_face_encoding_for_storage(stored_img)
                        if result["success"]:
                            stored_encodings = [result["encoding"]]

            if not stored_encodings:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message=f"No stored profile found for student {student_id}",
                    verification_type="storage"
                )

            preprocessed = self.image_preprocessor.preprocess_image(captured_image)
            if preprocessed is None:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="Failed to preprocess captured image - no face detected",
                    verification_type="preprocessing"
                )

            if preprocessed.dtype != np.uint8:
                if preprocessed.max() <= 1.0:
                    preprocessed = (preprocessed * 255).astype(np.uint8)
                else:
                    preprocessed = preprocessed.astype(np.uint8)

            captured_embedding = self._extract_embedding_direct(preprocessed)
            if captured_embedding is None:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="Failed to generate embedding from captured image",
                    verification_type="encoding"
                )

            similarities = []
            weights = self.encoding_weights.get(student_id, [1.0] * len(stored_encodings))
            
            for stored_encoding, weight in zip(stored_encodings, weights):
                stored_emb = np.array(stored_encoding)
                similarity = self._calculate_similarity(captured_embedding, stored_emb)
                weighted_similarity = similarity * weight
                similarities.append(weighted_similarity)

            best_similarity = max(similarities) if similarities else 0.0

            dynamic_threshold = self._get_dynamic_threshold(student_id)
            distance = 1.0 - best_similarity
            verified = distance <= dynamic_threshold
            
            self._update_student_threshold(student_id, best_similarity, verified)

            elapsed = time.time() - start_time
            if verified:
                self.metrics["successes"] += 1
                prev = self.metrics["successes"] - 1
                if prev > 0:
                    self.metrics["avg_time"] = ((self.metrics["avg_time"] * prev) + elapsed) / self.metrics["successes"]
                else:
                    self.metrics["avg_time"] = elapsed
            else:
                self.metrics["failures"] += 1

            return RecognitionResult(
                success=verified,
                confidence_score=best_similarity,
                verification_time=elapsed,
                verification_type="face",
                data={
                    "distance": distance,
                    "threshold_used": dynamic_threshold,
                    "liveness_score": live_result.get("score", 1.0),
                    "encodings_compared": len(stored_encodings),
                    "model_used": "Facenet"
                }
            )

        except Exception as e:
            logging.error(f"Verification error: {e}")
            self.metrics["failures"] += 1
            return RecognitionResult(
                success=False,
                error_message=str(e),
                verification_type="error"
            )

    def get_student_encoding(self, student_id: str) -> Optional[List]:
        """Get stored face encoding for a student"""
        if student_id in self.multiple_encodings:
            return self.multiple_encodings[student_id]
        return None

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        success_rate = 0.0
        if self.metrics["attempts"] > 0:
            success_rate = self.metrics["successes"] / self.metrics["attempts"]
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "cached_images": len(self.stored_images),
            "students_with_encodings": len(self.multiple_encodings),
            "dynamic_thresholds": len(self.student_thresholds),
            "model_type": "Facenet"
        }

    def verify_student_images(self, stored_image: np.ndarray, captured_image: np.ndarray) -> Dict:
        """Compare two face images directly with minimal file I/O"""
        try:
            stored_processed = self.image_preprocessor.preprocess_image(stored_image)
            captured_processed = self.image_preprocessor.preprocess_image(captured_image)
            
            if stored_processed is None or captured_processed is None:
                return {
                    "success": False,
                    "confidence_score": 0.0,
                    "message": "Failed to preprocess one or both images - no face detected"
                }
            
            if stored_processed.dtype != np.uint8:
                if stored_processed.max() <= 1.0:
                    stored_processed = (stored_processed * 255).astype(np.uint8)
                else:
                    stored_processed = stored_processed.astype(np.uint8)
                    
            if captured_processed.dtype != np.uint8:
                if captured_processed.max() <= 1.0:
                    captured_processed = (captured_processed * 255).astype(np.uint8)
                else:
                    captured_processed = captured_processed.astype(np.uint8)

            stored_embedding = self._extract_embedding_direct(stored_processed)
            captured_embedding = self._extract_embedding_direct(captured_processed)
            
            if stored_embedding is not None and captured_embedding is not None:
                similarity = self._calculate_similarity(stored_embedding, captured_embedding)
                distance = 1.0 - similarity
                
                return {
                    "success": distance <= Config.FACE_RECOGNITION_THRESHOLD,
                    "confidence_score": similarity,
                    "distance": distance,
                    "model_used": "Facenet",
                    "threshold": Config.FACE_RECOGNITION_THRESHOLD
                }
            else:
                return {
                    "success": False,
                    "confidence_score": 0.0,
                    "message": "Failed to generate face embeddings"
                }
                        
        except Exception as e:
            logging.error(f"Error comparing images: {str(e)}")
            return {
                "success": False,
                "confidence_score": 0.0,
                "message": str(e)
            }

    def register_multiple_face_angles(self, student_id: str, face_images: List[np.ndarray]) -> Dict:
        """Register multiple face angles for a student"""
        try:
            if not face_images:
                return {"success": False, "message": "No images provided"}
            
            successful_encodings = 0
            quality_scores = []
            failed_reasons = []
            
            for i, img in enumerate(face_images[:5]):
                result = self.get_face_encoding_for_storage(img, student_id)
                
                if result["success"]:
                    successful_encodings += 1
                    quality_scores.append(result.get("quality_score", 0.5))
                else:
                    failed_reasons.append(f"Image {i+1}: {result['message']}")
            
            if successful_encodings > 0:
                avg_quality = np.mean(quality_scores)
                return {
                    "success": True,
                    "message": f"Successfully registered {successful_encodings}/{len(face_images)} encodings",
                    "encodings_count": successful_encodings,
                    "average_quality": avg_quality,
                    "failed_reasons": failed_reasons if failed_reasons else None
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to register any face encodings",
                    "failed_reasons": failed_reasons
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Registration error: {str(e)}"
            }

    def get_student_verification_stats(self, student_id: str) -> Dict:
        """Get verification statistics for a specific student"""
        return {
            "student_id": student_id,
            "has_multiple_encodings": student_id in self.multiple_encodings,
            "encoding_count": len(self.multiple_encodings.get(student_id, [])),
            "average_encoding_quality": np.mean(self.encoding_weights.get(student_id, [0.5])),
            "dynamic_threshold": self._get_dynamic_threshold(student_id),
            "default_threshold": Config.FACE_RECOGNITION_THRESHOLD,
            "model_type": "Facenet"
        }

    def optimize_student_threshold(self, student_id: str, verification_history: List[Dict]):
        """Optimize threshold based on verification history"""
        if len(verification_history) < 5:
            return
        
        successful_similarities = [v["similarity"] for v in verification_history if v["verified"]]
        failed_similarities = [v["similarity"] for v in verification_history if not v["verified"]]
        
        if successful_similarities and failed_similarities:
            min_success = min(successful_similarities)
            max_fail = max(failed_similarities)
            optimal_threshold = 1.0 - ((min_success + max_fail) / 2)
            optimal_threshold = max(0.3, min(0.9, optimal_threshold))
            self.student_thresholds[student_id] = optimal_threshold
            self._save_multiple_encodings()

    def cleanup_old_encodings(self, days_old: int = 30):
        """Clean up old temporary files and optimize storage"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 3600)
            
            cleanup_count = 0
            for filename in os.listdir("."):
                if filename.startswith(('captured_image_', 'temp_preprocessed_', 'temp_face_')):
                    try:
                        if os.path.getctime(filename) < cutoff_time:
                            os.remove(filename)
                            cleanup_count += 1
                    except:
                        continue
            
            optimization_count = 0
            for student_id in list(self.multiple_encodings.keys()):
                if len(self.multiple_encodings[student_id]) > 5:
                    weights = self.encoding_weights.get(student_id, [])
                    if weights:
                        sorted_indices = np.argsort(weights)[::-1][:5]
                        self.multiple_encodings[student_id] = [
                            self.multiple_encodings[student_id][i] for i in sorted_indices
                        ]
                        self.encoding_weights[student_id] = [
                            self.encoding_weights[student_id][i] for i in sorted_indices
                        ]
                        optimization_count += 1
            
            self._save_multiple_encodings()
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

    def get_system_info(self) -> Dict:
        """Get system information and model status"""
        return {
            "model_info": {
                "facenet_loaded": True,
                "vggface_loaded": False,
                "model_type": "Facenet",
                "minimal_file_io": True
            },
            "storage_info": {
                "total_students": len(self.multiple_encodings),
                "total_stored_images": len(self.stored_images),
                "cached_encodings": sum(len(encodings) for encodings in self.multiple_encodings.values())
            },
            "performance_metrics": self.get_performance_metrics()
        }

    def batch_process_students(self, image_data_list: List[Dict]) -> List[Dict]:
        """Batch process multiple students for efficiency"""
        try:
            results = []
            
            for i, data in enumerate(image_data_list):
                student_id = data.get('student_id')
                image = data.get('image')
                
                if not student_id or image is None:
                    results.append({
                        "student_id": student_id,
                        "success": False,
                        "message": "Missing student_id or image"
                    })
                    continue
                
                result = self.verify_student(student_id, image)
                
                results.append({
                    "student_id": student_id,
                    "success": result.success,
                    "confidence_score": result.confidence_score,
                    "verification_time": result.verification_time,
                    "message": result.error_message if not result.success else "Verified successfully"
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            return [{
                "success": False,
                "message": f"Batch processing failed: {str(e)}"
            }]

    def export_student_data(self, student_id: str) -> Dict:
        """Export student face data for backup/transfer"""
        try:
            if student_id not in self.multiple_encodings:
                return {"success": False, "message": "Student not found"}
            
            export_data = {
                "student_id": student_id,
                "encodings": self.multiple_encodings[student_id],
                "weights": self.encoding_weights.get(student_id, []),
                "threshold": self.student_thresholds.get(student_id, Config.FACE_RECOGNITION_THRESHOLD),
                "model_type": "Facenet",
                "export_timestamp": time.time(),
                "encoding_count": len(self.multiple_encodings[student_id])
            }
            
            return {
                "success": True,
                "data": export_data,
                "message": f"Exported {len(self.multiple_encodings[student_id])} encodings"
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}

    def import_student_data(self, student_data: Dict) -> Dict:
        """Import student face data from backup"""
        try:
            required_fields = ["student_id", "encodings", "model_type"]
            for field in required_fields:
                if field not in student_data:
                    return {"success": False, "message": f"Missing required field: {field}"}
            
            student_id = student_data["student_id"]
            
            if student_data["model_type"] != "Facenet":
                return {"success": False, "message": "Incompatible model type"}
            
            self.multiple_encodings[student_id] = student_data["encodings"]
            self.encoding_weights[student_id] = student_data.get("weights", [1.0] * len(student_data["encodings"]))
            self.student_thresholds[student_id] = student_data.get("threshold", Config.FACE_RECOGNITION_THRESHOLD)
            
            self._save_multiple_encodings()
            
            return {
                "success": True,
                "message": f"Imported {len(student_data['encodings'])} encodings for {student_id}"
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}