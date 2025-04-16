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
import asyncio
import requests 
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, Optional, Set


# ===============================
# CONFIGURATION
# ===============================
class Config:
    """System configuration settings"""
    # Face Recognition Settings
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_RECOGNITION_THRESHOLD = 0.6
    IMAGE_SIZE = (224, 224)
    
    # Image Storage Paths
    TEMP_IMAGE_DIR = "temp_images/"
    STORED_IMAGES_DIR = "stored_images/"
    
    # Logging Settings
    LOG_FILE = "facial_recognition.log"
    LOG_LEVEL = "INFO"

# ===============================
# RESULT HANDLER
# ===============================
@dataclass
class RecognitionResult:
    """Enhanced recognition result structure"""
    success: bool
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    verification_time: Optional[float] = None
    verification_type: Optional[str] = None  # Added this field
    quality_details: Optional[Dict] = None
    data: Optional[Dict] = None

# ===============================
# ENUMS
# ===============================
class UserRole(Enum):
    """Defines user roles in the system"""
    TEACHER = "teacher"
    STUDENT = "student"

class AttendanceStatus(Enum):
    """Defines possible attendance statuses"""
    PRESENT = "present"
    ABSENT = "absent"
    PENDING = "pending"
    MANUALLY_MARKED = "manually_marked"

class SessionStatus(Enum):
    """Defines possible session statuses"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"

# ===============================
# DATA STRUCTURES
# ===============================
@dataclass
class User:
    """Base class for users"""
    id: str
    email: str
    password: str
    name: str
    role: UserRole
    last_login: Optional[str] = None
    is_active: bool = True

@dataclass
class Student:
    """Student user with face recognition data"""
    id: str
    email: str
    password: str
    face_encoding: Optional[List] = None
    registration_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: Optional[str] = None
    is_active: bool = True

@dataclass
class Teacher:
    """Teacher user with course assignments"""
    id: str
    email: str
    password: str
    name: str
    staff_id: str
    department: str
    role: UserRole = UserRole.TEACHER
    courses: List[str] = field(default_factory=list)
    last_login: Optional[str] = None
    is_active: bool = True

@dataclass
class Course:
    """Course information structure"""
    id: str
    name: str
    department: str
    teacher_id: str
    schedule: Dict = field(default_factory=dict)

@dataclass
class Session:
    """Attendance session details"""
    teacher_id: str
    hall_id: str
    start_time: str
    teacher_ip: str
    status: SessionStatus = SessionStatus.ACTIVE
    is_active: bool = True
    wifi_ssid: Optional[str] = None
    rssi_threshold: Optional[float] = None
    course_id: Optional[str] = None
    id: Optional[str] = None
    end_time: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: Optional[str] = None
    connected_students: Set[str] = field(default_factory=set)
    attendance_records: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class AttendanceRecord:
    """Individual attendance record"""
    id: str
    session_id: str
    student_id: str
    timestamp: str
    status: AttendanceStatus
    verification_details: Dict = field(default_factory=dict)
    modified_by: Optional[str] = None
    modification_reason: Optional[str] = None
    notification_sent: bool = False

@dataclass
class WifiSession:
    """Stores WiFi session information for attendance verification"""
    session_id: str
    teacher_id: str
    hall_id: str
    wifi_ssid: str  # Network name
    start_time: datetime
    is_active: bool = True
    connected_students: Set[str] = field(default_factory=set)  # Store connected student IDs

# ===============================
# LOGGING CONFIGURATION
# ===============================
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=Config.LOG_FILE
)

# ===============================
# CUSTOM EXCEPTIONS
# ===============================
class FaceRecognitionError(Exception):
    """Custom exception for face recognition errors"""
    pass

class SystemInitializationError(Exception):
    """Custom exception for system initialization errors"""
    pass

class CameraError(Exception):
    """Custom exception for camera-related errors"""
    pass

# ===============================
# PROGRESS INDICATOR CLASS
# ===============================
class ProgressIndicator:
    """Handles visual feedback for user interactions"""
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
    """Manages face encoding caching for better performance"""
    def __init__(self, cache_file: str = 'encodings_cache.pkl'):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logging.error(f"Cache loading error: {str(e)}")
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Cache saving error: {str(e)}")

    def get_encoding(self, image_path: str) -> Optional[List]:
        if image_path in self.cache:
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
    """Handles image preprocessing for face recognition"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = Config.IMAGE_SIZE) -> np.ndarray:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)

    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.3, beta: int = 5) -> np.ndarray:
        # Simple brightness and contrast adjustment
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def detect_and_align_face(image: np.ndarray) -> Optional[np.ndarray]:
        """Detects and aligns face in image with more lenient detection"""
        try:
            # Load cascade classifier
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple detection attempts with different parameters
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
            
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Add padding
            padding = 30
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            face_roi = image[y:y+h, x:x+w]
            
            # Simple quality check
            if not ImagePreprocessor.check_face_quality(face_roi):
                return None
            
            return ImagePreprocessor.resize_image(face_roi)
            
        except Exception as e:
            logging.error(f"Face detection error: {str(e)}")
            return None

    @staticmethod
    def check_face_quality(face_image: np.ndarray) -> bool:
        """Basic quality check with lenient thresholds"""
        try:
            # Basic size check
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return False
            
            # Convert to grayscale for checks
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Very lenient brightness check
            brightness = np.mean(gray)
            if brightness < 20 or brightness > 250:
                return False
            
            # Very lenient contrast check
            contrast = np.std(gray)
            if contrast < 10:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Face quality check error: {str(e)}")
            return False

    @staticmethod
    def preprocess_image(image: np.ndarray) -> Optional[np.ndarray]:
        """Simplified preprocessing pipeline"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            # Basic enhancement
            enhanced = ImagePreprocessor.adjust_brightness_contrast(image)
            
            # Detect face
            face_img = ImagePreprocessor.detect_and_align_face(enhanced)
            if face_img is None:
                return None

            # Final processing
            face_img = ImagePreprocessor.normalize_image(face_img)
            return face_img

        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            return None

# ===============================
# UTILITIES AND HELPERS
# ===============================
def check_requirements() -> bool:
    """Verifies all required packages are installed"""
    try:
        import cv2
        import numpy
        from deepface import DeepFace
        return True
    except ImportError as e:
        ProgressIndicator.show_error(f"Missing requirement: {str(e)}")
        return False

def setup_directories():
    """Creates necessary system directories"""
    try:
        directories = [
            Config.TEMP_IMAGE_DIR,
            Config.STORED_IMAGES_DIR,
            'logs'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                ProgressIndicator.show_status(f"Created directory: {directory}")
    except Exception as e:
        raise SystemInitializationError(f"Directory setup failed: {str(e)}")

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
        with camera_context() as cap:
            if not cap.isOpened():
                raise CameraError("Failed to open webcam")

            ProgressIndicator.show_status("Camera ready. Position your face in the frame...")
            
            cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise CameraError("Failed to capture frame")

                preview_frame = frame.copy()
                height, width = preview_frame.shape[:2]
                
                center_x = width // 2
                center_y = height // 2
                size = min(width, height) // 3
                
                cv2.rectangle(preview_frame, 
                            (center_x - size, center_y - size),
                            (center_x + size, center_y + size),
                            (0, 255, 0), 2)

                cv2.putText(preview_frame, "Position face within the green box", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview_frame, "Press SPACE to capture or Q to quit", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Camera Preview', preview_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return None, None
                elif key == ord(' '):
                    roi = frame[center_y-size:center_y+size, center_x-size:center_x+size]
                    if roi.size > 0:
                        break
                    else:
                        ProgressIndicator.show_warning("Invalid capture region. Please try again.")
                        continue

            cv2.destroyAllWindows()
            
            # Save the captured image temporarily
            filename = f"captured_image_{int(time.time())}.jpg"
            cv2.imwrite(filename, roi)
            
            ProgressIndicator.show_success(f"Image captured successfully")
            return filename, roi.copy()

    except Exception as e:
        ProgressIndicator.show_error(f"Capture error: {str(e)}")
        logging.error(f"Capture error: {str(e)}")
        return None, None
    finally:
        cv2.destroyAllWindows()

def generate_encoding(image_path: str) -> Optional[List]:
    """Generates face encoding from image"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise FaceRecognitionError("Cannot read image")

        preprocessed_img = ImagePreprocessor.preprocess_image(img)
        if preprocessed_img is None:
            raise FaceRecognitionError("Face preprocessing failed")

        temp_path = f"temp_preprocessed_{os.path.basename(image_path)}"
        cv2.imwrite(temp_path, (preprocessed_img * 255).astype(np.uint8))

        encoding = DeepFace.represent(img_path=temp_path, model_name="Facenet")
        os.remove(temp_path)
        return encoding

    except Exception as e:
        logging.error(f"Encoding error: {str(e)}")
        return None

def cleanup_temp_files(keep_files=False):
    """Cleans up temporary system files"""
    if keep_files:
        return

    try:
        # Clean main directory
        for file in os.listdir():
            if file.startswith(('captured_image_', 'temp_preprocessed_')):
                os.remove(file)
                
        # Clean temp directory
        if os.path.exists(Config.TEMP_IMAGE_DIR):
            for file in os.listdir(Config.TEMP_IMAGE_DIR):
                if file.startswith('temp_'):
                    os.remove(os.path.join(Config.TEMP_IMAGE_DIR, file))
    except Exception as e:
        logging.error(f"Cleanup error: {str(e)}")

def get_device_network_info() -> Dict:
    """Get current device's network information"""
    try:
        import socket
        
        # Get hostname and IP
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        return {
            "hostname": hostname,
            "ip_address": ip_address
        }
    except Exception as e:
        logging.error(f"Error getting network info: {str(e)}")
        return {
            "error": str(e)
        }
# ===============================
# Manages attendance sessions and network verification
# ===============================   
@dataclass
class Session:
    """Session data structure"""
    teacher_id: str
    hall_id: str
    start_time: str
    teacher_ip: str
    is_active: bool = True
    connected_students: Set[str] = field(default_factory=set)
    attendance_records: Dict[str, Dict] = field(default_factory=dict)

class AttendanceSession:
    """Enhanced attendance session management"""
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}

    def start_session(self, teacher_id: str, hall_id: str) -> Dict:
        """Start new teaching session with enhanced tracking"""
        try:
            # Get network info
            network_info = get_device_network_info()
            if "error" in network_info:
                return {
                    "success": False,
                    "message": "Failed to get network information",
                    "error_type": "network"
                }

            # Generate session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hall_id}"

            # Create new session with dataclass
            self.active_sessions[session_id] = Session(
                teacher_id=teacher_id,
                hall_id=hall_id,
                start_time=datetime.now().isoformat(),
                teacher_ip=network_info["ip_address"],
                is_active=True  # Explicitly set session as active
            )

            logging.info(f"Session created: {session_id}")
            print(f"Active sessions: {list(self.active_sessions.keys())}")  # Debug line

            return {
                "success": True,
                "message": "Session started successfully",
                "session_id": session_id,
                "network_info": network_info
            }
        except Exception as e:
            logging.error(f"Error starting session: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to start session: {str(e)}",
                "error_type": "system"
            }

    def verify_student_network(self, session_id: str) -> Dict:
        """Enhanced network verification"""
        try:
            print(f"Verifying session: {session_id}")  # Debug line
            print(f"Available sessions: {list(self.active_sessions.keys())}")  # Debug line

            if session_id not in self.active_sessions:
                return {
                    "success": False,
                    "message": "No active session found",
                    "error_type": "session"
                }

            session = self.active_sessions[session_id]
            if not session.is_active:
                return {
                    "success": False,
                    "message": "Session has ended",
                    "error_type": "session"
                }

            # Get and verify student's network
            student_network = get_device_network_info()
            if "error" in student_network:
                return {
                    "success": False,
                    "message": "Failed to get student's network information",
                    "error_type": "network"
                }

            # Enhanced network comparison
            student_subnet = student_network["ip_address"].split('.')[:3]
            teacher_subnet = session.teacher_ip.split('.')[:3]
            
            if student_subnet != teacher_subnet:
                return {
                    "success": False,
                    "message": "Not connected to the same network as teacher",
                    "error_type": "network",
                    "details": {
                        "student_subnet": '.'.join(student_subnet),
                        "teacher_subnet": '.'.join(teacher_subnet)
                    }
                }

            # Add student to connected students
            session.connected_students.add(student_network["ip_address"])

            return {
                "success": True,
                "message": "Network verification successful",
                "student_ip": student_network["ip_address"],
                "teacher_ip": session.teacher_ip
            }

        except Exception as e:
            logging.error(f"Error verifying network: {str(e)}")
            return {
                "success": False,
                "message": f"Network verification failed: {str(e)}",
                "error_type": "system"
            }

    def mark_attendance(self, session_id: str, student_id: str, 
                       verification_result: Dict) -> Dict:
        """Record student attendance"""
        try:
            if session_id not in self.active_sessions:
                return {
                    "success": False,
                    "message": "Session not found"
                }

            session = self.active_sessions[session_id]
            if not session.is_active:
                return {
                    "success": False,
                    "message": "Session has ended"
                }

            # Record attendance
            session.attendance_records[student_id] = {
                "timestamp": datetime.now().isoformat(),
                "face_confidence": verification_result.get("face_confidence"),
                "wifi_verified": verification_result.get("wifi_verified"),
                "status": "present"
            }

            return {
                "success": True,
                "message": "Attendance marked successfully",
                "data": session.attendance_records[student_id]
            }

        except Exception as e:
            logging.error(f"Error marking attendance: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to mark attendance: {str(e)}"
            }

    def end_session(self, session_id: str, teacher_id: str) -> Dict:
        """Enhanced session ending with attendance summary"""
        try:
            if session_id not in self.active_sessions:
                return {
                    "success": False,
                    "message": "Session not found"
                }
            
            session = self.active_sessions[session_id]
            if session.teacher_id != teacher_id:
                return {
                    "success": False,
                    "message": "Unauthorized to end this session"
                }
            
            session.is_active = False
            
            # Generate session summary
            summary = {
                "total_students": len(session.attendance_records),
                "connected_devices": len(session.connected_students),
                "start_time": session.start_time,
                "end_time": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "message": "Session ended successfully",
                "summary": summary
            }
        except Exception as e:
            logging.error(f"Error ending session: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to end session: {str(e)}"
            }
        

class FaceQualityChecker:
    """Enhanced face quality checking system"""
    
    @staticmethod
    def check_lighting(image: np.ndarray) -> Dict:
        """Check image lighting conditions"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            return {
                "success": True,
                "brightness": brightness,
                "contrast": contrast,
                "is_good_lighting": (40 <= brightness <= 240 and contrast >= 20)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def check_face_position(image: np.ndarray) -> Dict:
        """Check face position and orientation"""
        try:
            # Load face detection model
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
                return {
                    "success": True,
                    "face_detected": False,
                    "message": "No face detected"
                }
            
            if len(faces) > 1:
                return {
                    "success": True,
                    "face_detected": False,
                    "message": "Multiple faces detected"
                }
            
            # Get face position
            x, y, w, h = faces[0]
            center_x = x + w/2
            center_y = y + h/2
            
            # Check if face is centered
            img_center_x = image.shape[1]/2
            img_center_y = image.shape[0]/2
            
            is_centered = (
                abs(center_x - img_center_x) < image.shape[1]/4 and
                abs(center_y - img_center_y) < image.shape[0]/4
            )
            
            return {
                "success": True,
                "face_detected": True,
                "is_centered": is_centered,
                "face_position": {"x": x, "y": y, "width": w, "height": h}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def check_face_quality(image: np.ndarray) -> Dict:
        """Comprehensive face quality check"""
        try:
            # Check image size
            if image.shape[0] < 100 or image.shape[1] < 100:
                return {
                    "success": True,
                    "is_quality_good": False,
                    "message": "Image resolution too low"
                }
            
            # Check lighting
            lighting = FaceQualityChecker.check_lighting(image)
            if not lighting["success"]:
                return {
                    "success": False,
                    "error": lighting["error"]
                }
            
            # Check face position
            position = FaceQualityChecker.check_face_position(image)
            if not position["success"]:
                return {
                    "success": False,
                    "error": position["error"]
                }
            
            # Compile results
            is_quality_good = (
                lighting["is_good_lighting"] and
                position["face_detected"] and
                position["is_centered"]
            )
            
            return {
                "success": True,
                "is_quality_good": is_quality_good,
                "details": {
                    "lighting": lighting,
                    "position": position
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def enhance_face_recognition(self, image: np.ndarray) -> Dict:
    """Enhanced face recognition with quality checks"""
    try:
        # First check image quality
        quality_result = FaceQualityChecker.check_face_quality(image)
        if not quality_result["success"]:
            return {
                "success": False,
                "message": f"Quality check failed: {quality_result['error']}"
            }
            
        if not quality_result["is_quality_good"]:
            return {
                "success": False,
                "message": "Image quality not sufficient for recognition",
                "details": quality_result["details"]
            }
            
        # Proceed with face recognition
        preprocessed = self.image_preprocessor.preprocess_image(image)
        if preprocessed is None:
            return {
                "success": False,
                "message": "Failed to preprocess image"
            }
            
        # Generate encoding
        encoding_result = self.get_face_encoding_for_storage(preprocessed)
        if not encoding_result["success"]:
            return {
                "success": False,
                "message": encoding_result["message"]
            }
            
        return {
            "success": True,
            "encoding": encoding_result["encoding"],
            "quality_details": quality_result["details"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Face recognition error: {str(e)}"
        }
# ===============================
# WiFi verification system
# ===============================
class WifiVerificationSystem:
    """
    Handles WiFi verification for attendance system
    Note: This is the ML/verification part only. 
    Backend team needs to implement the actual network checking logic.
    """
    
    def __init__(self):
        self.active_sessions = {}  # Dictionary to store active sessions
        
    def create_session(self, teacher_id: str, hall_id: str, wifi_ssid: str) -> Dict:
        """
        Create a new teaching session
        This function will be called by the backend when a teacher starts a class
        """
        try:
            # Generate unique session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hall_id}"
            
            # Create new session
            session = WifiSession(
                session_id=session_id,
                teacher_id=teacher_id,
                hall_id=hall_id,
                wifi_ssid=wifi_ssid,
                start_time=datetime.now()
            )
            
            # Store session
            self.active_sessions[session_id] = session
            
            return {
                "success": True,
                "message": "Teaching session created successfully",
                "session_id": session_id,
                "wifi_ssid": wifi_ssid
            }
            
        except Exception as e:
            logging.error(f"Error creating teaching session: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create session: {str(e)}"
            }

    def verify_wifi_connection(self, session_id: str, student_id: str, 
                             student_wifi_data: Dict) -> Dict:
        """
        Verify if student is connected to the correct WiFi network
        student_wifi_data will be provided by the backend team
        """
        try:
            # Check if session exists and is active
            if session_id not in self.active_sessions:
                return {
                    "success": False,
                    "message": "No active teaching session found"
                }
            
            session = self.active_sessions[session_id]
            
            # Check if session is still active
            if not session.is_active:
                return {
                    "success": False,
                    "message": "Teaching session has ended"
                }
            
            # Verify SSID matches
            if student_wifi_data.get("ssid") != session.wifi_ssid:
                return {
                    "success": False,
                    "message": "Not connected to the correct WiFi network"
                }
            
            # Add student to connected students
            session.connected_students.add(student_id)
            
            return {
                "success": True,
                "message": "WiFi verification successful",
                "session_id": session_id,
                "verification_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error verifying student WiFi: {str(e)}")
            return {
                "success": False,
                "message": f"WiFi verification failed: {str(e)}"
            }

    def end_session(self, session_id: str, teacher_id: str) -> Dict:
        """End a teaching session"""
        try:
            if session_id not in self.active_sessions:
                return {
                    "success": False,
                    "message": "Session not found"
                }
            
            session = self.active_sessions[session_id]
            if session.teacher_id != teacher_id:
                return {
                    "success": False,
                    "message": "Unauthorized to end this session"
                }
            
            session.is_active = False
            
            return {
                "success": True,
                "message": "Session ended successfully",
                "session_data": {
                    "session_id": session_id,
                    "start_time": session.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "connected_students": len(session.connected_students)
                }
            }
            
        except Exception as e:
            logging.error(f"Error ending session: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to end session: {str(e)}"
            }
# ===============================
# FACE RECOGNITION SYSTEM
# ===============================
class FaceRecognitionSystem:
    """Main face recognition system handling registration and verification"""
    def __init__(self):
        try:
            ProgressIndicator.show_status("Initializing Face Recognition System...")
            self.encoding_cache = EncodingCache()
            self.image_preprocessor = ImagePreprocessor()
            self.stored_images = self._load_stored_images()
            self._cache_stored_images()
            ProgressIndicator.show_success("System initialized successfully")
        except Exception as e:
            raise SystemInitializationError(f"System initialization failed: {str(e)}")

    def _load_stored_images(self) -> List[str]:
        """Loads stored student images from directory"""
        try:
            if not os.path.exists(Config.STORED_IMAGES_DIR):
                os.makedirs(Config.STORED_IMAGES_DIR)
            images = [os.path.join(Config.STORED_IMAGES_DIR, f) 
                     for f in os.listdir(Config.STORED_IMAGES_DIR) 
                     if f.endswith('.jpg')]
            ProgressIndicator.show_status(f"Found {len(images)} stored images")
            return images
        except Exception as e:
            logging.error(f"Error loading stored images: {str(e)}")
            return []

    def _cache_stored_images(self):
        """Pre-caches encodings for all stored images"""
        total = len(self.stored_images)
        for idx, image in enumerate(self.stored_images, 1):
            ProgressIndicator.show_status(
                f"Processing image {idx}/{total}: {image}...",
                end='\r'
            )
            self.encoding_cache.get_encoding(image)
        print()
        ProgressIndicator.show_success("All images cached successfully")

    def get_face_encoding_for_storage(self, image: np.ndarray) -> Dict:
        """
        Generates face encoding in a format suitable for database storage
        """
        temp_path = None
        try:
            # Preprocess the image
            preprocessed = self.image_preprocessor.preprocess_image(image)
            if preprocessed is None:
                return {
                    "success": False,
                    "encoding": None,
                    "message": "Failed to preprocess image",
                    "timestamp": datetime.now().isoformat()
                }

            # Save temporary image
            temp_path = os.path.join(Config.TEMP_IMAGE_DIR, f"temp_{int(time.time())}.jpg")
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))

            # Generate encoding
            encoding = DeepFace.represent(img_path=temp_path, model_name="Facenet")

            if encoding is None:
                return {
                    "success": False,
                    "encoding": None,
                    "message": "Failed to generate encoding",
                    "timestamp": datetime.now().isoformat()
                }

            # Convert encoding to list for JSON serialization
            encoding_list = encoding[0]['embedding']
            if isinstance(encoding_list, np.ndarray):
                encoding_list = encoding_list.tolist()

            return {
                "success": True,
                "encoding": encoding_list,
                "message": "Encoding generated successfully",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Encoding generation error: {str(e)}")
            return {
                "success": False,
                "encoding": None,
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logging.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    logging.error(f"Error cleaning up temporary file: {str(e)}")

    def verify_student(self, student_id: str, captured_image: np.ndarray) -> RecognitionResult:
        """Verifies student identity using facial recognition"""
        try:
            # Get stored encoding
            stored_encoding = self.get_student_encoding(student_id)
            if stored_encoding is None:
                return RecognitionResult(
                    success=False,
                    error_message="No stored encoding found",
                    verification_type="storage"
                )

            # Generate new encoding
            captured_encoding = self.generate_live_encoding(captured_image)
            if captured_encoding is None:
                return RecognitionResult(
                    success=False,
                    error_message="Failed to generate encoding",
                    verification_type="encoding"
                )

            # Compare encodings
            start_time = time.time()
            verification_result = DeepFace.verify(
                img1_path=captured_encoding[0]['embedding'],
                img2_path=stored_encoding[0]['embedding'],
                model_name="Facenet"
            )
            verification_time = time.time() - start_time

            return RecognitionResult(
                success=verification_result.get('verified', False),
                confidence_score=verification_result.get('distance', 0),
                verification_time=verification_time,
                verification_type="face",
                data={
                    "timestamp": datetime.now().isoformat(),
                    "verification_details": verification_result
                }
            )

        except Exception as e:
            logging.error(f"Verification error: {str(e)}")
            return RecognitionResult(
                success=False,
                error_message=str(e),
                verification_type="error"
            )
        finally:
            # Cleanup: Remove any temporary files
            self.cleanup_temp_files()

    def get_student_encoding(self, student_id: str) -> Optional[List]:
        """Retrieves stored encoding for a student"""
        try:
            image_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No image found for student {student_id}")
            return self.encoding_cache.get_encoding(image_path)
        except Exception as e:
            logging.error(f"Error getting student encoding: {str(e)}")
            return None

    def generate_live_encoding(self, image: np.ndarray) -> Optional[List]:
        """Generates encoding for live captured image"""
        temp_path = None
        try:
            preprocessed = self.image_preprocessor.preprocess_image(image)
            if preprocessed is None:
                return None
                
            temp_path = os.path.join(Config.TEMP_IMAGE_DIR, f"temp_verify_{int(time.time())}.jpg")
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))
            
            encoding = generate_encoding(temp_path)
            return encoding
            
        except Exception as e:
            logging.error(f"Error generating live encoding: {str(e)}")
            return None
        finally:
            # Cleanup: Remove temporary image
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logging.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    logging.error(f"Error cleaning up temporary file: {str(e)}")

    def register_new_student(self, student_id: str, image: np.ndarray) -> Dict:
        """Registers new student with facial encoding"""
        try:
            # Basic image checks
            if image is None or image.size == 0:
                return {
                    "success": False,
                    "message": "Invalid image data",
                    "encoding": None
                }

            # Get encoding for storage
            encoding_result = self.get_face_encoding_for_storage(image)
            if not encoding_result["success"]:
                return {
                    "success": False,
                    "message": encoding_result["message"],
                    "encoding": None
                }

            # Resize and preprocess image
            resized_image = cv2.resize(image, Config.IMAGE_SIZE)
            preprocessed_image = cv2.convertScaleAbs(resized_image, alpha=1.3, beta=5)

            # Save the preprocessed image
            save_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
            cv2.imwrite(save_path, preprocessed_image)

            # Update stored images list
            if save_path not in self.stored_images:
                self.stored_images.append(save_path)

            return {
                "success": True,
                "message": "Student registered successfully",
                "encoding": encoding_result["encoding"],
                "timestamp": encoding_result["timestamp"]
            }

        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}",
                "encoding": None
            }

    def process_verification(self, captured_image_path: str) -> Dict:
        """Processes general verification against all stored images"""
        try:
            # First, read and check the captured image
            img = cv2.imread(captured_image_path)
            if img is None:
                return {
                    "status": "error",
                    "message": "Failed to read captured image"
                }

            # Check face quality first
            if not ImagePreprocessor.check_face_quality(img):
                return {
                    "status": "error",
                    "message": "Image quality too low. Please ensure good lighting and clear face view"
                }

            # Preprocess the image before generating encoding
            preprocessed_img = self.image_preprocessor.preprocess_image(img)
            if preprocessed_img is None:
                return {
                    "status": "error",
                    "message": "Failed to detect face in image. Please ensure face is clearly visible"
                }

            # Save preprocessed image temporarily
            temp_preprocessed_path = os.path.join(
                Config.TEMP_IMAGE_DIR, 
                f"temp_preprocessed_{int(time.time())}.jpg"
            )
            cv2.imwrite(temp_preprocessed_path, (preprocessed_img * 255).astype(np.uint8))

            try:
                # Generate encoding with more specific error handling
                new_encoding = DeepFace.represent(
                    img_path=temp_preprocessed_path,
                    model_name="Facenet",
                    enforce_detection=True
                )

                if new_encoding is None:
                    raise FaceRecognitionError("Failed to generate encoding from preprocessed image")

                matches = []
                confidence_scores = {}
                
                # Compare with stored images
                for stored_image in self.stored_images:
                    try:
                        stored_encoding = self.encoding_cache.get_encoding(stored_image)
                        if stored_encoding is None:
                            logging.warning(f"Skipping comparison with {stored_image}: No valid encoding")
                            continue

                        result = DeepFace.verify(
                            img1_path=new_encoding[0]['embedding'],
                            img2_path=stored_encoding[0]['embedding'],
                            model_name="Facenet",
                            distance_metric="cosine"  # Add specific distance metric
                        )

                        if result.get('verified', False):
                            matches.append(stored_image)
                            confidence_scores[stored_image] = result.get('distance', 0)

                    except Exception as e:
                        logging.error(f"Error comparing with {stored_image}: {str(e)}")
                        continue

                return {
                    "status": "success",
                    "matches": matches,
                    "confidence_scores": confidence_scores,
                    "total_comparisons": len(self.stored_images),
                    "successful_comparisons": len(matches)
                }

            finally:
                # Clean up temporary preprocessed image
                if os.path.exists(temp_preprocessed_path):
                    os.remove(temp_preprocessed_path)

        except Exception as e:
            logging.error(f"Verification error: {str(e)}")
            return {
                "status": "error",
                "message": f"Verification failed: {str(e)}. Please try again with better lighting and clear face view"
            }
        finally:
            # Cleanup temporary files
            self.cleanup_temp_files()
            # Remove the captured image
            if os.path.exists(captured_image_path):
                try:
                    os.remove(captured_image_path)
                    logging.debug(f"Removed captured image: {captured_image_path}")
                except Exception as e:
                    logging.error(f"Error removing captured image: {str(e)}")
    def cleanup_temp_files(self):
        """Cleans up all temporary files created during verification"""
        try:
            # Clean temp directory
            if os.path.exists(Config.TEMP_IMAGE_DIR):
                for file in os.listdir(Config.TEMP_IMAGE_DIR):
                    if file.startswith('temp_verify_'):
                        file_path = os.path.join(Config.TEMP_IMAGE_DIR, file)
                        try:
                            os.remove(file_path)
                            logging.debug(f"Cleaned up: {file_path}")
                        except Exception as e:
                            logging.error(f"Error removing temp file {file_path}: {str(e)}")

            # Clean main directory
            current_dir = os.getcwd()
            for file in os.listdir(current_dir):
                if file.startswith(('captured_image_', 'temp_verify_')):
                    try:
                        os.remove(file)
                        logging.debug(f"Cleaned up: {file}")
                    except Exception as e:
                        logging.error(f"Error removing file {file}: {str(e)}")

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
    def verify_attendance_with_wifi(self, student_id: str, captured_image: np.ndarray,
                                session_id: str, wifi_data: Dict,
                                wifi_system: WifiVerificationSystem) -> Dict:
        """
        Combined verification of face and WiFi
        This is the main function that combines both face and WiFi verification
        """
        try:
            # First verify WiFi connection
            wifi_result = wifi_system.verify_wifi_connection(
                session_id=session_id,
                student_id=student_id,
                student_wifi_data=wifi_data
            )
            
            if not wifi_result["success"]:
                return {
                    "success": False,
                    "message": f"WiFi verification failed: {wifi_result['message']}",
                    "verification_type": "wifi"
                }

            # Then verify face
            face_result = self.verify_student(student_id, captured_image)
            if not face_result.success:
                return {
                    "success": False,
                    "message": f"Face verification failed: {face_result.error_message}",
                    "verification_type": "face"
                }

            return {
                "success": True,
                "message": "Attendance verification successful",
                "face_confidence": face_result.confidence_score,
                "wifi_verified": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Attendance verification error: {str(e)}")
            return {
                "success": False,
                "message": f"Verification failed: {str(e)}",
                "verification_type": "system"
            }    

# ===============================
# MAIN EXECUTION
# ===============================
class NetworkInfoDisplay:
    """Handles network information display"""
    @staticmethod
    def display_network_info():
        print("\n--- Current Network Information ---")
        network_info = get_device_network_info()
        
        if "error" not in network_info:
            print(f"Hostname: {network_info['hostname']}")
            print(f"IP Address: {network_info['ip_address']}")
            print("-" * 40)
            return network_info
        else:
            ProgressIndicator.show_warning(f"Could not get network info: {network_info['error']}")
            return None
class MenuHandler:
    @staticmethod
    def display_main_menu():
        print("\n=== Attendance System Menu ===")
        print("1. Start New Session")
        print("2. Verify Attendance")
        print("3. End Session")
        print("4. Test Verification")
        print("5. Register New Student")
        print("6. Exit")

    @staticmethod
    def display_registration_instructions():
        print("\n=== Student Registration Instructions ===")
        print("1. Fill in the required student information")
        print("2. When ready to capture face image:")
        print("   - Ensure good lighting")
        print("   - Look directly at the camera")
        print("   - Keep a neutral expression")
        print("3. Press SPACE to capture or Q to quit")

    @staticmethod
    def display_verification_instructions():
        print("\n=== Face Verification Instructions ===")
        print("1. Position your face within the green box")
        print("2. Ensure good lighting conditions")
        print("3. Look directly at the camera")
        print("4. Keep a neutral expression")
        print("5. Stay still during capture")
        print("6. Press SPACE to capture or Q to quit")

    @staticmethod
    def get_user_input(prompt: str) -> str:
        return input(prompt).strip()
 
class StudentRegistrationHandler:
    def __init__(self, face_recognition_system: FaceRecognitionSystem):
        self.system = face_recognition_system

    async def handle_registration(self) -> Dict:
        """Handle new student registration process"""
        try:
            print("\n=== New Student Registration ===")
            
            # Display instructions and capture face image first
            MenuHandler.display_registration_instructions()
            image_path, captured_image = capture_image()

            if image_path is None or captured_image is None:
                return {
                    "success": False,
                    "message": "Image capture cancelled"
                }

            try:
                # Check image quality
                quality_result = FaceQualityChecker.check_face_quality(captured_image)
                if not quality_result["success"] or not quality_result.get("is_quality_good", False):
                    return {
                        "success": False,
                        "message": "Image quality not sufficient for registration",
                        "details": quality_result.get("details", {})
                    }

                # Check if face already exists in database
                verification_result = self.system.process_verification(image_path)
                if verification_result["status"] == "success" and verification_result["matches"]:
                    return {
                        "success": False,
                        "message": "Face already registered in the system. Cannot create duplicate account.",
                        "details": {
                            "matches_found": len(verification_result["matches"]),
                            "confidence_scores": verification_result["confidence_scores"]
                        }
                    }

                # If face is not registered, proceed with getting user information
                student_info = self._get_student_info()
                if not student_info["success"]:
                    return student_info

                # Generate unique student ID based on timestamp
                student_id = f"STU_{int(time.time())}"

                # Register student with face image
                registration_result = self.system.register_new_student(
                    student_id,
                    captured_image
                )

                if registration_result["success"]:
                    return {
                        "success": True,
                        "message": "Student registered successfully",
                        "student_id": student_id,
                        "details": {
                            "email": student_info["email"],
                            "registration_time": datetime.now().isoformat()
                        }
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Registration failed: {registration_result['message']}"
                    }

            finally:
                # Cleanup captured image
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        logging.error(f"Error removing captured image: {str(e)}")

        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}"
            }

    def _get_student_info(self) -> Dict:
        """Get student email and password"""
        try:
            email = MenuHandler.get_user_input("Enter Email: ")
            password = MenuHandler.get_user_input("Enter Password: ")

            if not all([email, password]):
                return {
                    "success": False,
                    "message": "Email and password are required"
                }

            # Basic email validation
            if not '@' in email or not '.' in email:
                return {
                    "success": False,
                    "message": "Invalid email format"
                }

            # Basic password validation
            if len(password) < 6:
                return {
                    "success": False,
                    "message": "Password must be at least 6 characters long"
                }

            return {
                "success": True,
                "email": email,
                "password": password
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting student information: {str(e)}"
            }  

class SessionHandler:
    def __init__(self, system, attendance_session, wifi_system):
        self.system = system
        self.attendance_session = attendance_session
        self.wifi_system = wifi_system

    async def handle_start_session(self) -> Dict:
        """Handle starting a new session"""
        print("\n=== Start New Teaching Session ===")
        teacher_id = MenuHandler.get_user_input("Enter Teacher ID: ")
        hall_id = MenuHandler.get_user_input("Enter Hall ID: ")
        
        # Use start_session instead of create_session
        session_result = self.attendance_session.start_session(
            teacher_id=teacher_id,
            hall_id=hall_id
        )
        
        if session_result["success"]:
            # Create WiFi session
            wifi_result = self.wifi_system.create_session(
                teacher_id=teacher_id,
                hall_id=hall_id,
                wifi_ssid=get_device_network_info().get('ip_address', 'unknown')
            )
            
            if wifi_result["success"]:
                return {
                    "success": True,
                    "session_id": session_result["session_id"],
                    "message": "Session started successfully",
                    "network_info": session_result.get("network_info", {})
                }
            
        return {
            "success": False,
            "message": session_result.get("message", "Failed to start session")
        }

    async def handle_end_session(self) -> Dict:
        """Handle ending a session"""
        print("\n=== End Teaching Session ===")
        session_id = MenuHandler.get_user_input("Enter Session ID: ")
        teacher_id = MenuHandler.get_user_input("Enter Teacher ID: ")
        
        return self.attendance_session.end_session(session_id, teacher_id)
class VerificationHandler:
    """Handles all verification-related operations"""
    
    def __init__(self, system, attendance_session, wifi_system):
        self.system = system
        self.attendance_session = attendance_session
        self.wifi_system = wifi_system

    def handle_verification_failure(self, student_id: str, error_type: str, details: str = "") -> Dict:
        """Handle verification failures with specific recovery options"""
        error_response = {
            "success": False,
            "student_id": student_id,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }

        if error_type == "wifi":
            error_response.update({
                "message": "WiFi verification failed",
                "recovery_options": [
                    "Reconnect to classroom WiFi",
                    "Verify you are in the correct classroom",
                    "Check if the session is still active",
                    "Contact teacher for manual verification"
                ]
            })
        elif error_type == "face":
            error_response.update({
                "message": "Face verification failed",
                "recovery_options": [
                    "Ensure good lighting",
                    "Remove face coverings",
                    "Face the camera directly",
                    "Try again with better positioning"
                ]
            })
        elif error_type == "session":
            error_response.update({
                "message": "Session verification failed",
                "recovery_options": [
                    "Verify session ID is correct",
                    "Check if session is still active",
                    "Ask teacher to verify session status"
                ]
            })
        elif error_type == "quality":
            error_response.update({
                "message": "Image quality insufficient",
                "recovery_options": [
                    "Find better lighting",
                    "Ensure clear face view",
                    "Remove any obstructions",
                    "Keep still during capture"
                ]
            })
        elif error_type == "input":
            error_response.update({
                "message": "Invalid input data",
                "recovery_options": [
                    "Verify all required fields",
                    "Check input format",
                    "Try again with correct information"
                ]
            })
        elif error_type == "capture":
            error_response.update({
                "message": "Image capture failed",
                "recovery_options": [
                    "Check camera connection",
                    "Ensure camera permissions",
                    "Try again with better lighting",
                    "Restart the application"
                ]
            })
        else:
            error_response.update({
                "message": f"Verification failed: {details}",
                "recovery_options": [
                    "Try again",
                    "Contact system administrator"
                ]
            })

        # Display error message and recovery options
        ProgressIndicator.show_error(f"\n{error_response['message']}")
        print("\nRecovery Options:")
        for i, option in enumerate(error_response['recovery_options'], 1):
            print(f"{i}. {option}")

        return error_response

    def handle_successful_verification(self, student_id: str, 
                                    session_id: str, 
                                    verification_result: RecognitionResult) -> Dict:
        """Handle successful verification process"""
        return {
            "success": True,
            "message": "Verification successful",
            "data": {
                "student_id": student_id,
                "session_id": session_id,
                "confidence_score": verification_result.confidence_score,
                "verification_time": verification_result.verification_time,
                "verification_type": verification_result.verification_type,
                "timestamp": datetime.now().isoformat()
            }
        }

    async def handle_verification(self) -> Dict:
        """Handle student verification process"""
        print("\n=== Student Attendance Verification ===")
        
        try:
            # Get student and session info
            student_id = MenuHandler.get_user_input("Enter Student ID: ")
            session_id = MenuHandler.get_user_input("Enter Session ID: ")

            if not all([student_id, session_id]):
                return self.handle_verification_failure(
                    student_id if 'student_id' in locals() else "unknown",
                    "input",
                    "Missing required fields"
                )

            # Verify network
            network_result = self.attendance_session.verify_student_network(session_id)
            if not network_result["success"]:
                return self.handle_verification_failure(
                    student_id,
                    network_result.get("error_type", "network"),
                    network_result["message"]
                )

            # Verify student exists in system
            if not os.path.exists(os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")):
                return self.handle_verification_failure(
                    student_id,
                    "input",
                    "Student ID not found in system"
                )

            # Capture and verify face
            MenuHandler.display_verification_instructions()
            image_path, img = capture_image()

            if image_path is None or img is None:
                return self.handle_verification_failure(
                    student_id,
                    "capture",
                    "Image capture cancelled"
                )

            try:
                # Check image quality first
                quality_result = FaceQualityChecker.check_face_quality(img)
                if not quality_result["success"] or not quality_result.get("is_quality_good", False):
                    return self.handle_verification_failure(
                        student_id,
                        "quality",
                        quality_result.get("message", "Image quality check failed")
                    )

                # Verify student
                result = self.system.verify_student(student_id, img)
                
                if result.success:
                    # Mark attendance if verification successful
                    attendance_result = self.handle_successful_verification(
                        student_id, session_id, result
                    )
                    
                    # Mark attendance in session
                    mark_result = self.attendance_session.mark_attendance(
                        session_id,
                        student_id,
                        {
                            "face_confidence": result.confidence_score,
                            "wifi_verified": True
                        }
                    )
                    
                    if not mark_result["success"]:
                        return self.handle_verification_failure(
                            student_id,
                            "attendance",
                            mark_result["message"]
                        )
                    
                    return attendance_result
                
                return self.handle_verification_failure(
                    student_id,
                    "face",
                    result.error_message
                )

            finally:
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        logging.error(f"Error removing captured image: {str(e)}")

        except Exception as e:
            logging.error(f"Verification process error: {str(e)}")
            return self.handle_verification_failure(
                student_id if 'student_id' in locals() else "unknown",
                "system",
                str(e)
            )

    async def handle_test_verification(self) -> Dict:
        """Handle test verification process"""
        try:
            print("\n=== Test Verification Mode ===")
            
            # Get network information first
            network_info = get_device_network_info()
            if "error" in network_info:
                return self.handle_verification_failure(
                    "test",
                    "network",
                    "Failed to get network information for testing"
                )

            # Create test session
            test_session_result = self.wifi_system.create_session(
                teacher_id="TEST_TEACHER",
                hall_id="TEST_HALL",
                wifi_ssid=network_info["ip_address"]
            )

            if not test_session_result["success"]:
                return self.handle_verification_failure(
                    "test",
                    "session",
                    f"Failed to create test session: {test_session_result['message']}"
                )

            session_id = test_session_result["session_id"]
            ProgressIndicator.show_success(
                f"Test session created successfully\n"
                f"Session ID: {session_id}\n"
                f"Network: {network_info['ip_address']}"
            )

            # Get student ID for testing
            student_id = MenuHandler.get_user_input("Enter student ID for testing: ")
            
            if not student_id:
                return self.handle_verification_failure(
                    "test",
                    "input",
                    "Student ID is required"
                )

            # Verify student exists in system
            if not os.path.exists(os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")):
                return self.handle_verification_failure(
                    student_id,
                    "input",
                    "Student ID not found in system"
                )

            # Prepare test WiFi data
            wifi_data = {
                "ssid": network_info["ip_address"],
                "timestamp": datetime.now().isoformat()
            }

            # Display capture instructions and get image
            MenuHandler.display_verification_instructions()
            image_path, img = capture_image()

            if image_path is None or img is None:
                return self.handle_verification_failure(
                    student_id,
                    "capture",
                    "Image capture cancelled"
                )

            try:
                # First test face quality
                quality_result = FaceQualityChecker.check_face_quality(img)
                if not quality_result["success"] or not quality_result.get("is_quality_good", False):
                    return self.handle_verification_failure(
                        student_id,
                        "quality",
                        "Image quality check failed"
                    )

                # Perform combined verification
                verification_result = self.system.verify_attendance_with_wifi(
                    student_id=student_id,
                    captured_image=img,
                    session_id=session_id,
                    wifi_data=wifi_data,
                    wifi_system=self.wifi_system
                )

                if verification_result["success"]:
                    return {
                        "success": True,
                        "message": "Test verification successful",
                        "data": {
                            "student_id": student_id,
                            "session_id": session_id,
                            "face_confidence": verification_result.get("face_confidence"),
                            "wifi_verified": verification_result.get("wifi_verified"),
                            "timestamp": verification_result.get("timestamp"),
                            "test_details": {
                                "network_info": network_info,
                                "quality_metrics": quality_result["details"]
                            }
                        }
                    }
                
                return self.handle_verification_failure(
                    student_id,
                    verification_result.get("verification_type", "unknown"),
                    verification_result.get("message", "Verification failed")
                )

            finally:
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        logging.error(f"Error removing test image: {str(e)}")

        except Exception as e:
            logging.error(f"Test verification error: {str(e)}")
            return self.handle_verification_failure(
                "test",
                "system",
                str(e)
            )
async def main():
    """Enhanced main execution function"""
    try:
        # System initialization
        ProgressIndicator.show_status("Initializing system...")
        
        if not check_requirements():
            ProgressIndicator.show_error("System requirements not met")
            return
        
        setup_directories()
        
        # Initialize systems
        system = FaceRecognitionSystem()
        attendance_session = AttendanceSession()
        wifi_system = WifiVerificationSystem()
        
        # Initialize handlers
        session_handler = SessionHandler(system, attendance_session, wifi_system)
        verification_handler = VerificationHandler(system, attendance_session, wifi_system)
        registration_handler = StudentRegistrationHandler(system) 

        while True:
            try:
                # Display menu and get network info
                MenuHandler.display_main_menu()
                network_info = NetworkInfoDisplay.display_network_info()
                choice = MenuHandler.get_user_input("Select an option (1-6): ")
                
                if choice == "1":
                    result = await session_handler.handle_start_session()
                    if result["success"]:
                        ProgressIndicator.show_success(
                            f"Session started successfully\n"
                            f"Session ID: {result['session_id']}\n"
                            f"Network: {network_info['ip_address'] if network_info else 'Not available'}"
                        )
                    else:
                        ProgressIndicator.show_error(result["message"])
                
                elif choice == "2":
                    if not network_info:
                        ProgressIndicator.show_error("Network connection required for verification")
                        continue
                        
                    result = await verification_handler.handle_verification()
                    if result["success"]:
                        ProgressIndicator.show_success(
                            "\nVerification Results:\n" +
                            "-" * 40 +
                            f"\nStatus: Successful" +
                            f"\nConfidence: {result['data']['confidence_score']:.3f}" +
                            f"\nTime: {result['data']['timestamp']}\n" +
                            f"Network: {network_info['ip_address']}"
                        )
                    else:
                        ProgressIndicator.show_error(result["message"])
                
                elif choice == "3":
                    result = await session_handler.handle_end_session()
                    if result["success"]:
                        ProgressIndicator.show_success(
                            "Session ended successfully\n" +
                            f"Network: {network_info['ip_address'] if network_info else 'Not available'}"
                        )
                        if "summary" in result:
                            print("\nSession Summary:")
                            print(f"Total Students: {result['summary']['total_students']}")
                            print(f"Connected Devices: {result['summary']['connected_devices']}")
                            print(f"Duration: {result['summary']['start_time']} - {result['summary']['end_time']}")
                    else:
                        ProgressIndicator.show_error(result["message"])
                
                elif choice == "4":
                    result = await verification_handler.handle_test_verification()
                    if result["success"]:
                        ProgressIndicator.show_success(
                            "\nTest Verification Results:\n" +
                            "-" * 40 +
                            f"\nStatus: Successful" +
                            f"\nFace Confidence: {result.get('face_confidence', 0):.3f}" +
                            f"\nWiFi Verified: {result.get('wifi_verified', False)}" +
                            f"\nTimestamp: {result.get('timestamp', 'N/A')}"
                        )
                        
                        # Display detailed test metrics
                        if "test_details" in result:
                            print("\nTest Details:")
                            print("-" * 40)
                            print(f"Network: {result['test_details']['network_info']['ip_address']}")
                            
                            quality_metrics = result['test_details']['quality_metrics']
                            if 'lighting' in quality_metrics:
                                print(f"Brightness: {quality_metrics['lighting'].get('brightness', 'N/A'):.1f}")
                                print(f"Contrast: {quality_metrics['lighting'].get('contrast', 'N/A'):.1f}")
                            
                            if 'position' in quality_metrics:
                                pos = quality_metrics['position']
                                print(f"Face Detected: {pos.get('face_detected', False)}")
                                print(f"Face Centered: {pos.get('is_centered', False)}")
                    else:
                        ProgressIndicator.show_error(
                            f"\nTest Verification Failed:" +
                            f"\nReason: {result['message']}"
                        )
                        
                        # Show troubleshooting tips
                        print("\nTroubleshooting Tips:")
                        if result.get('verification_type') == 'wifi':
                            print("1. Check network connection")
                            print("2. Verify you're on the same network")
                            print("3. Try reconnecting to the network")
                        elif result.get('verification_type') == 'face':
                            print("1. Ensure good lighting")
                            print("2. Face the camera directly")
                            print("3. Remove any face coverings")
                            print("4. Try adjusting your position")

                elif choice == "5":
                    # Handle student registration
                    result = await registration_handler.handle_registration()
                    if result["success"]:
                        ProgressIndicator.show_success(
                            f"\nStudent Registration Successful!" +
                            f"\nStudent ID: {result['student_id']}" +
                            f"\nEmail: {result['details']['email']}" +
                            f"\nRegistration Time: {result['details']['registration_time']}"
                        )
                    else:
                        ProgressIndicator.show_error(result["message"])
                        if "details" in result and "matches_found" in result["details"]:
                            print("\nRegistration blocked: Face already registered")
                            print(f"Number of matches found: {result['details']['matches_found']}")
                        
                elif choice == "6":
                    ProgressIndicator.show_status("Cleaning up and shutting down...")
                    break
                
                else:
                    ProgressIndicator.show_warning("Invalid choice. Please select 1-5.")
                
                # Add a small delay between operations
                await asyncio.sleep(1)
                    
            except Exception as e:
                ProgressIndicator.show_error(f"Operation error: {str(e)}")
                logging.error(f"Operation error: {str(e)}")
                await asyncio.sleep(2)  # Give user time to read error
                
    except KeyboardInterrupt:
        ProgressIndicator.show_status("\nSystem interrupted by user")
    except Exception as e:
        ProgressIndicator.show_error(f"System error: {str(e)}")
        logging.error(f"System error: {str(e)}")
    finally:
        try:
            cleanup_temp_files()
            ProgressIndicator.show_status("System shutdown complete")
        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=Config.LOG_FILE
    )
    
    # Reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Run the main application
    asyncio.run(main())