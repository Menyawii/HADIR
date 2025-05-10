import cv2
import numpy as np
import time
import os
from typing import Tuple, Optional
from contextlib import contextmanager
from core.utils.config import Config
from core.utils.exceptions import CameraError
from demo.ui.progress_indicator import ProgressIndicator

@contextmanager
def camera_context():
    """Manages camera resources with improved error handling"""
    cap = None
    try:
        # Try with DirectShow backend first, then fall back to default
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            raise CameraError("Failed to initialize camera")
            
        # Give camera time to initialize
        time.sleep(0.5)
        
        # Set safer camera parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        yield cap
    except Exception as e:
        raise CameraError(f"Camera error: {str(e)}")
    finally:
        if cap is not None:
            # Release camera resources properly
            for _ in range(3):  # Multiple release attempts
                try:
                    cap.release()
                    break
                except:
                    time.sleep(0.1)
        cv2.destroyAllWindows()

def capture_image() -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Capture image from webcam with improved stability and error handling"""
    cap = None
    try:
        # Try different backends and indices in order of preference
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),  # Often works best on Windows
            (cv2.CAP_ANY, "Default"),       # System default
            (cv2.CAP_MSMF, "Media Foundation")  # Windows Media Foundation
        ]
        
        # Try to find a working camera
        for backend, name in backends:
            for index in [0, 1]:  # Usually only need to try 0 and 1
                try:
                    print(f"Trying {name} backend with camera index {index}")
                    cap = cv2.VideoCapture(index, backend)
                    
                    if cap.isOpened():
                        # Verify we can read from it
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"Success with {name} backend at index {index}")
                            
                            # Set safer camera parameters
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            
                            # Break out of both loops with a working camera
                            break
                        else:
                            print(f"Camera opened but couldn't read frames")
                            if cap:
                                cap.release()
                                cap = None
                    else:
                        print(f"Failed to open with {name} at index {index}")
                        if cap:
                            cap.release()
                            cap = None
                except Exception as e:
                    print(f"Error with {name} at index {index}: {str(e)}")
                    if cap:
                        cap.release()
                        cap = None
            
            # If we found a working camera, stop trying backends
            if cap is not None and cap.isOpened():
                break
                
        # If we couldn't find any working camera
        if cap is None or not cap.isOpened():
            raise CameraError("Failed to open webcam - no working camera found")
        
        # Small initialization delay
        time.sleep(0.5)
        ProgressIndicator.show_status("Camera ready. Position your face in the frame...")
        
        # Create window
        cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
        
        # Camera capture loop with error recovery
        consecutive_errors = 0
        frame = None
        
        while True:
            try:
                ret, new_frame = cap.read()
                
                if not ret or new_frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        raise CameraError("Multiple consecutive frame capture failures")
                    time.sleep(0.1)  # Short delay before retry
                    continue
                
                # Reset error counter on success
                consecutive_errors = 0
                frame = new_frame  # Keep last good frame
                
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
                    # Capture the region of interest with bounds checking
                    y1 = max(0, center_y - size)
                    y2 = min(height, center_y + size)
                    x1 = max(0, center_x - size)
                    x2 = min(width, center_x + size)
                    
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        break
                    else:
                        ProgressIndicator.show_warning(
                            "Invalid capture region. Please try again."
                        )
                        continue
                        
            except Exception as frame_error:
                print(f"Frame error: {str(frame_error)}")
                consecutive_errors += 1
                if consecutive_errors > 5:
                    raise CameraError(f"Camera operation failed: {str(frame_error)}")
                time.sleep(0.1)

        # Clean up properly
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
        # Ensure camera is properly released
        if 'cap' in locals() and cap is not None:
            try:
                # Multiple release attempts to ensure cleanup
                for _ in range(3):
                    try:
                        cap.release()
                        break
                    except:
                        time.sleep(0.1)
            except:
                pass
            
        # Ensure all windows are closed
        for _ in range(2):
            cv2.destroyAllWindows()
            time.sleep(0.1)