import cv2
import numpy as np
from typing import Dict
import logging

class LivenessDetector:
    """Lightweight liveness detection using Haar cascades, frame differencing, and texture analysis."""
    
    def __init__(self):
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if self.eye_cascade.empty():
            logging.warning("Failed to load eye cascade classifier.")

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Analyze liveness of a captured image using eye detection, motion, and texture.
        Returns a dict with keys: 'live', 'score', and 'explanation'.
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                return {
                    "live": False,
                    "score": 0.0,
                    "explanation": "Invalid image input",
                    "message": "Invalid image input"  # Add message field
                }

            # 1) Texture score via Laplacian variance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_score = min(float(lap_var) / 1000.0, 1.0)  # Convert to float

            # 2) Eye detection (blink proxy)
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            eye_score = 1.0 if len(eyes) >= 2 else 0.0

            # 3) Motion detection via two-frame differencing
            motion_score = 0.0
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret1, f1 = cap.read()
                    time.sleep(0.3)
                    ret2, f2 = cap.read()
                    cap.release()
                    if ret1 and ret2:
                        diff = cv2.absdiff(f1, f2)
                        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        non_zero = cv2.countNonZero(gray_diff)
                        # if >2% of pixels change, count as motion
                        if non_zero > 0.02 * gray_diff.size:
                            motion_score = 1.0
            except Exception:
                pass
            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

            # 4) Combine with weights
            score = float(0.4 * texture_score + 0.3 * eye_score + 0.3 * motion_score)
            live = bool(score >= 0.5)  # Convert to standard bool

            # 5) Build explanation
            explanation = (
                f"Eyes detected: {'Yes' if eye_score>0 else 'No'}; "
                f"Motion: {'Yes' if motion_score>0 else 'No'}; "
                f"Texture: {texture_score:.2f}"
            )

            return {
                "live": live,
                "score": score,
                "explanation": explanation,
                "message": "Liveness check failed" if not live else "Liveness check passed"
            }

        except Exception as e:
            logging.error(f"Liveness detection error: {str(e)}")
            return {
                "live": False,
                "score": 0.0,
                "explanation": f"Error during liveness detection: {str(e)}",
                "message": "Liveness detection failed"
            }