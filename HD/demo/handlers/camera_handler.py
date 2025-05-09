import cv2
import numpy as np
import time
from typing import Tuple, Optional
from contextlib import contextmanager
from core.utils.config import Config
from core.utils.exceptions import CameraError
from demo.ui.progress_indicator import ProgressIndicator

@contextmanager
def camera_context():
    """Manages camera resources"""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise CameraError("Failed to initialize camera")
        yield cap
    except Exception as e:
        raise CameraError(f"Camera error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def capture_image() -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Capture image from webcam with preview"""
    try:
        # Try different camera indices
        camera_indices = [0, 1, 2]  # Add more if needed
        cap = None
        
        for index in camera_indices:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                break
            cap.release()
            
        if cap is None or not cap.isOpened():
            raise CameraError("Failed to open webcam - no working camera found")

        ProgressIndicator.show_status("Camera ready. Position your face in the frame...")
        
        # Rest of your code remains the same...
        
        cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                raise CameraError("Failed to capture frame")

            preview_frame = frame.copy()
            height, width = preview_frame.shape[:2]
            
            # Draw guide box
            center_x = width // 2
            center_y = height // 2
            size = min(width, height) // 3
            
            cv2.rectangle(preview_frame, 
                        (center_x - size, center_y - size),
                        (center_x + size, center_y + size),
                        (0, 255, 0), 2)

            # Add instructions
            cv2.putText(preview_frame, "Position face within the green box", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview_frame, "Press SPACE to capture or Q to quit", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Camera Preview', preview_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, None
            elif key == ord(' '):
                # Capture the region of interest
                roi = frame[center_y-size:center_y+size, 
                          center_x-size:center_x+size]
                if roi.size > 0:
                    break
                else:
                    ProgressIndicator.show_warning(
                        "Invalid capture region. Please try again."
                    )
                    continue

        cv2.destroyAllWindows()
        
        # Save captured image
        filename = f"captured_image_{int(time.time())}.jpg"
        cv2.imwrite(filename, roi)
        
        ProgressIndicator.show_success(f"Image captured successfully")
        return filename, roi.copy()

    except Exception as e:
        ProgressIndicator.show_error(f"Capture error: {str(e)}")
        return None, None
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()