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
import glob
import asyncio
import requests 
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple, Union, Any, Set
from enum import Enum

# ===============================
# CONFIGURATION
# ===============================
@dataclass
class Config:
    """System configuration settings"""
    # Face Recognition Settings
    FACE_DETECTION_CONFIDENCE: float = 0.9
    FACE_RECOGNITION_THRESHOLD: float = 0.6
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    
    # Image Storage Paths
    TEMP_IMAGE_DIR: str = "temp_images/"
    STORED_IMAGES_DIR: str = "stored_images/"
    
    # Logging Settings
    LOG_FILE: str = "facial_recognition.log"
    LOG_LEVEL: str = "INFO"

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
    verification_type: Optional[str] = None
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
class BaseUser:
    """Base class for all users"""
    id: str
    email: str
    password: str
    name: str
    is_active: bool = True
    last_login: Optional[str] = None

@dataclass
class Student(BaseUser):
    """Student user with face recognition data"""
    role: UserRole = UserRole.STUDENT
    face_encoding: Optional[List] = None
    registration_date: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Teacher(BaseUser):
    """Teacher user with course assignments"""
    staff_id: str
    department: str
    role: UserRole = UserRole.TEACHER
    courses: List[str] = field(default_factory=list)

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
    """Unified session structure for attendance and WiFi"""
    id: str
    teacher_id: str
    hall_id: str
    start_time: str
    teacher_ip: str
    wifi_ssid: Optional[str] = None
    rssi_threshold: Optional[float] = None
    course_id: Optional[str] = None
    end_time: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    is_active: bool = True
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
class SystemError(Exception):
    """Base exception for system errors"""
    pass

class FaceRecognitionError(SystemError):
    """Custom exception for face recognition errors"""
    pass

class SystemInitializationError(SystemError):
    """Custom exception for system initialization errors"""
    pass

class CameraError(SystemError):
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
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def check_face_quality(face_image: np.ndarray) -> Dict[str, Any]:
        """Unified face quality check"""
        try:
            quality_result = {
                "success": False,
                "brightness": 0,
                "contrast": 0,
                "size_ok": False
            }

            # Size check
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return quality_result
            quality_result["size_ok"] = True

            # Convert to grayscale for checks
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            # Brightness and contrast check
            quality_result["brightness"] = np.mean(gray)
            quality_result["contrast"] = np.std(gray)
            
            # Quality validation
            quality_result["success"] = (
                quality_result["size_ok"] and
                20 < quality_result["brightness"] < 250 and
                quality_result["contrast"] > 10
            )
            
            return quality_result
            
        except Exception as e:
            logging.error(f"Face quality check error: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def detect_and_align_face(image: np.ndarray) -> Optional[np.ndarray]:
        """Detects and aligns face in image"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple detection attempts
            for scale in [1.1, 1.2, 1.3]:
                for min_neighbors in [3, 4, 5]:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(50, 50),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(faces) > 0:
                        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                        padding = 30
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(image.shape[1] - x, w + 2*padding)
                        h = min(image.shape[0] - y, h + 2*padding)
                        
                        face_roi = image[y:y+h, x:x+w]
                        return ImagePreprocessor.resize_image(face_roi)
            
            return None
            
        except Exception as e:
            logging.error(f"Face detection error: {str(e)}")
            return None

    @staticmethod
    def preprocess_image(image: np.ndarray) -> Optional[np.ndarray]:
        """Unified preprocessing pipeline"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            enhanced = ImagePreprocessor.adjust_brightness_contrast(image)
            face_img = ImagePreprocessor.detect_and_align_face(enhanced)
            
            if face_img is None:
                return None

            return ImagePreprocessor.normalize_image(face_img)

        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            return None
        
# ===============================
# UTILITIES AND HELPERS
# ===============================
def check_requirements() -> bool:
    """Verifies all required packages are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'deepface': 'deepface'
    }
    
    try:
        for module, package in required_packages.items():
            __import__(module)
        return True
    except ImportError as e:
        ProgressIndicator.show_error(f"Missing requirement: {package} ({str(e)})")
        return False

def setup_directories():
    """Creates necessary system directories"""
    directories = [
        Config.TEMP_IMAGE_DIR,
        Config.STORED_IMAGES_DIR,
        'logs'
    ]
    
    try:
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
            ProgressIndicator.show_status("Camera ready. Position your face in the frame...")
            
            cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise CameraError("Failed to capture frame")

                preview_frame = frame.copy()
                height, width = preview_frame.shape[:2]
                
                # Setup capture guide
                center_x, center_y = width // 2, height // 2
                size = min(width, height) // 3
                
                # Draw guide box and instructions
                cv2.rectangle(preview_frame, 
                            (center_x - size, center_y - size),
                            (center_x + size, center_y + size),
                            (0, 255, 0), 2)
                
                instructions = [
                    "Position face within the green box",
                    "Press SPACE to capture or Q to quit"
                ]
                
                for idx, text in enumerate(instructions):
                    cv2.putText(preview_frame, text, 
                              (10, 30 + idx * 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)

                cv2.imshow('Camera Preview', preview_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return None, None
                elif key == ord(' '):
                    roi = frame[center_y-size:center_y+size, 
                              center_x-size:center_x+size]
                    if roi.size > 0:
                        break
                    else:
                        ProgressIndicator.show_warning("Invalid capture region. Please try again.")

            cv2.destroyAllWindows()
            
            filename = f"captured_image_{int(time.time())}.jpg"
            cv2.imwrite(filename, roi)
            
            ProgressIndicator.show_success("Image captured successfully")
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

        try:
            encoding = DeepFace.represent(img_path=temp_path, model_name="Facenet")
            return encoding
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logging.error(f"Encoding error: {str(e)}")
        return None

def get_device_network_info() -> Dict[str, str]:
    """Get current device's network information"""
    try:
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        return {
            "hostname": hostname,
            "ip_address": ip_address,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error getting network info: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def cleanup_temp_files(keep_files: bool = False):
    """Cleans up temporary system files"""
    if keep_files:
        return

    try:
        # Define patterns for cleanup
        temp_patterns = [
            ('', ['captured_image_*', 'temp_preprocessed_*']),
            (Config.TEMP_IMAGE_DIR, ['temp_*'])
        ]
        
        for directory, patterns in temp_patterns:
            for pattern in patterns:
                path = os.path.join(directory, pattern) if directory else pattern
                for file in glob.glob(path):
                    try:
                        os.remove(file)
                    except Exception as e:
                        logging.warning(f"Failed to remove {file}: {str(e)}")
    except Exception as e:
        logging.error(f"Cleanup error: {str(e)}")

# ===============================
# SESSION MANAGEMENT SYSTEM
# ===============================   
class SessionManager:
    """Unified session management system"""
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}

    def _generate_session_id(self, hall_id: str) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hall_id}"

    def start_session(self, teacher_id: str, hall_id: str) -> Dict:
        """Start new teaching session with network verification"""
        try:
            # Get network info
            network_info = get_device_network_info()
            if "error" in network_info:
                return {
                    "success": False,
                    "message": "Failed to get network information",
                    "error_type": "network"
                }

            session_id = self._generate_session_id(hall_id)
            
            # Create unified session
            self.active_sessions[session_id] = Session(
                id=session_id,
                teacher_id=teacher_id,
                hall_id=hall_id,
                start_time=datetime.now().isoformat(),
                teacher_ip=network_info["ip_address"],
                wifi_ssid=network_info["ip_address"],  # Using IP as SSID for verification
                is_active=True
            )

            logging.info(f"Session created: {session_id}")

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

    def verify_student_network(self, session_id: str, student_ip: Optional[str] = None) -> Dict:
        """Verify student's network connection"""
        try:
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

            # Get student's network info if not provided
            if not student_ip:
                student_network = get_device_network_info()
                if "error" in student_network:
                    return {
                        "success": False,
                        "message": "Failed to get student's network information",
                        "error_type": "network"
                    }
                student_ip = student_network["ip_address"]

            # Compare network subnets
            student_subnet = student_ip.split('.')[:3]
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
            session.connected_students.add(student_ip)

            return {
                "success": True,
                "message": "Network verification successful",
                "student_ip": student_ip,
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

            # Record attendance with timestamp
            attendance_record = {
                "timestamp": datetime.now().isoformat(),
                "face_confidence": verification_result.get("face_confidence"),
                "wifi_verified": verification_result.get("wifi_verified", False),
                "status": AttendanceStatus.PRESENT.value
            }

            session.attendance_records[student_id] = attendance_record
            session.modified_at = datetime.now().isoformat()

            return {
                "success": True,
                "message": "Attendance marked successfully",
                "data": attendance_record
            }

        except Exception as e:
            logging.error(f"Error marking attendance: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to mark attendance: {str(e)}"
            }

    def end_session(self, session_id: str, teacher_id: str) -> Dict:
        """End session with attendance summary"""
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
            
            # Update session status
            session.is_active = False
            session.status = SessionStatus.COMPLETED
            session.end_time = datetime.now().isoformat()
            session.modified_at = session.end_time
            
            # Generate session summary
            summary = {
                "total_students": len(session.attendance_records),
                "connected_devices": len(session.connected_students),
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": self._calculate_duration(session.start_time, session.end_time)
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

    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """Calculate session duration"""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        except Exception:
            return "Duration calculation failed"       
# ===============================
# FACE QUALITY CHECKER
# ===============================
class FaceQualityChecker:
    """Handles face quality assessment"""
    
    @staticmethod
    def check_face_quality(image: np.ndarray) -> Dict:
        """Comprehensive face quality check"""
        try:
            # Check image size
            if image.shape[0] < 100 or image.shape[1] < 100:
                return {
                    "success": False,
                    "message": "Image resolution too low",
                    "details": {
                        "width": image.shape[1],
                        "height": image.shape[0]
                    }
                }
            
            # Check lighting
            lighting = FaceQualityChecker.check_lighting(image)
            if not lighting["success"]:
                return lighting
            
            # Check face position
            position = FaceQualityChecker.check_face_position(image)
            if not position["success"]:
                return position
            
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
            return {
                "success": False,
                "message": f"Quality check failed: {str(e)}",
                "error": str(e)
            }

    @staticmethod
    def check_lighting(image: np.ndarray) -> Dict:
        """Check image lighting conditions"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            is_good_lighting = (40 <= brightness <= 240 and contrast >= 20)
            
            return {
                "success": True,
                "brightness": brightness,
                "contrast": contrast,
                "is_good_lighting": is_good_lighting
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Lighting check failed: {str(e)}",
                "error": str(e)
            }

    @staticmethod
    def check_face_position(image: np.ndarray) -> Dict:
        """Check face position and orientation"""
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
                "face_position": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Face position check failed: {str(e)}",
                "error": str(e)
            }

# ===============================
# FACE RECOGNITION SYSTEM
# ===============================
class FaceRecognitionSystem:
    """Unified face recognition and quality checking system"""
    def __init__(self, dataset_path='dataset', attendance_path='attendance'):
        self.dataset_path = dataset_path
        self.attendance_path = attendance_path
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.attendance_path, exist_ok=True)

    def register_student(self, student_id: str, student_name: str, images: List[np.ndarray]) -> Dict:
        """Register a new student with face images"""
        try:
            student_folder = os.path.join(self.dataset_path, student_id)
            os.makedirs(student_folder, exist_ok=True)
            
            saved_images = []
            for idx, image in enumerate(images):
                # Check image quality first
                quality_result = FaceQualityChecker.check_face_quality(image)
                if not quality_result["success"]:
                    continue

                image_path = os.path.join(student_folder, f"{student_name}_{idx}.jpg")
                cv2.imwrite(image_path, image)
                saved_images.append(image_path)

            if not saved_images:
                return {
                    "success": False,
                    "message": "No quality images provided for registration"
                }

            return {
                "success": True,
                "message": f"Registered {student_name} with ID {student_id}",
                "saved_images": len(saved_images)
            }

        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}"
            }

    def verify_student(self, captured_image: np.ndarray, student_id: str) -> RecognitionResult:
        """Verify student identity"""
        try:
            student_folder = os.path.join(self.dataset_path, student_id)
            if not os.path.exists(student_folder):
                return RecognitionResult(
                    success=False,
                    error_message=f"No data found for student ID {student_id}"
                )

            # Save captured image temporarily
            tmp_capture_path = f"temp_capture_{int(time.time())}.jpg"
            cv2.imwrite(tmp_capture_path, captured_image)

            try:
                for file in os.listdir(student_folder):
                    if not file.endswith((".jpg", ".png")):
                        continue
                        
                    db_image_path = os.path.join(student_folder, file)
                    try:
                        start_time = time.time()
                        result = DeepFace.verify(
                            tmp_capture_path, 
                            db_image_path,
                            enforce_detection=False
                        )
                        verification_time = time.time() - start_time

                        if result["verified"]:
                            return RecognitionResult(
                                success=True,
                                confidence_score=result.get("distance", 0),
                                verification_time=verification_time,
                                verification_type="face"
                            )
                    except Exception as e:
                        logging.warning(f"Verification attempt failed: {str(e)}")
                        continue

                return RecognitionResult(
                    success=False,
                    error_message="No matching face found",
                    verification_type="face"
                )

            finally:
                if os.path.exists(tmp_capture_path):
                    os.remove(tmp_capture_path)

        except Exception as e:
            logging.error(f"Verification error: {str(e)}")
            return RecognitionResult(
                success=False,
                error_message=str(e),
                verification_type="system"
            )

    def verify_attendance_with_wifi(
        self, 
        student_id: str, 
        captured_image: np.ndarray,
        session_id: str, 
        wifi_data: Dict,
        session_manager: 'SessionManager'
    ) -> Dict:
        """Combined face and WiFi verification"""
        try:
            # Verify face first
            face_result = self.verify_student(captured_image, student_id)
            
            # Verify WiFi
            wifi_result = session_manager.verify_student_network(
                session_id, 
                wifi_data.get("ip_address")
            )

            if face_result.success and wifi_result["success"]:
                # Mark attendance
                attendance_data = {
                    "face_confidence": face_result.confidence_score,
                    "wifi_verified": True,
                    "verification_time": face_result.verification_time,
                    "verification_type": "combined"
                }
                
                return {
                    "success": True,
                    "message": "Attendance verified and marked",
                    "data": attendance_data,
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "success": False,
                "message": "Verification failed",
                "face_verified": face_result.success,
                "wifi_verified": wifi_result["success"],
                "error_type": "wifi" if face_result.success else "face",
                "details": {
                    "face_error": face_result.error_message,
                    "wifi_error": wifi_result.get("message")
                }
            }

        except Exception as e:
            logging.error(f"Attendance verification error: {str(e)}")
            return {
                "success": False,
                "message": f"Verification failed: {str(e)}",
                "error_type": "system"
            }

    def process_verification(self, captured_image_path: str) -> Dict:
        """Process verification request"""
        try:
            image = cv2.imread(captured_image_path)
            if image is None:
                return {
                    "success": False,
                    "message": "Image not found or cannot be read"
                }

            # Check image quality first
            quality_result = FaceQualityChecker.check_face_quality(image)
            if not quality_result["success"]:
                return {
                    "success": False,
                    "message": "Image quality insufficient",
                    "details": quality_result
                }

            results = DeepFace.find(
                image, 
                db_path=self.dataset_path, 
                enforce_detection=False
            )

            if results and len(results[0]) > 0:
                top_match = results[0].iloc[0]
                identity_path = top_match["identity"]
                student_id = identity_path.split(os.sep)[-2]
                return {
                    "success": True,
                    "student_id": student_id,
                    "confidence": 1 - top_match["distance"],
                    "matches": True
                }

            return {
                "success": True,
                "message": "No match found",
                "matches": False
            }

        except Exception as e:
            logging.error(f"Verification processing error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "error_type": "processing"
            }
# ===============================
# DISPLAY AND MENU SYSTEM
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
    """Handles menu display and instructions"""
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

    @staticmethod
    def display_result(result: Dict, operation_type: str):
        """Unified method for displaying operation results"""
        if result["success"]:
            ProgressIndicator.show_success(f"\n{operation_type} Successful!")
            if "data" in result:
                print("\nDetails:")
                for key, value in result["data"].items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            ProgressIndicator.show_error(f"\n{operation_type} Failed!")
            print(f"Reason: {result.get('message', 'Unknown error')}")
            if "details" in result:
                print("\nError Details:")
                for key, value in result["details"].items():
                    print(f"{key.replace('_', ' ').title()}: {value}")

# ===============================
# SYSTEM HANDLERS
# ===============================
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
                if verification_result.get("matches", False):
                    return {
                        "success": False,
                        "message": "Face already registered in the system",
                        "details": verification_result
                    }

                # Get student information
                student_info = self._get_student_info()
                if not student_info["success"]:
                    return student_info

                # Generate unique student ID
                student_id = f"STU_{int(time.time())}"

                # Register student with face image
                registration_result = self.system.register_student(
                    student_id,
                    student_info["email"].split('@')[0],  # Use email username as name
                    [captured_image]
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
                    return registration_result

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
    def __init__(self, system: FaceRecognitionSystem, session_manager: SessionManager):
        self.system = system
        self.session_manager = session_manager

    async def handle_start_session(self) -> Dict:
        """Handle starting a new session"""
        print("\n=== Start New Teaching Session ===")
        teacher_id = MenuHandler.get_user_input("Enter Teacher ID: ")
        hall_id = MenuHandler.get_user_input("Enter Hall ID: ")
        
        return self.session_manager.start_session(
            teacher_id=teacher_id,
            hall_id=hall_id
        )

    async def handle_end_session(self) -> Dict:
        """Handle ending a session"""
        print("\n=== End Teaching Session ===")
        session_id = MenuHandler.get_user_input("Enter Session ID: ")
        teacher_id = MenuHandler.get_user_input("Enter Teacher ID: ")
        
        return self.session_manager.end_session(session_id, teacher_id)

class VerificationHandler:
    """Handles all verification-related operations"""
    
    def __init__(self, system: FaceRecognitionSystem, session_manager: SessionManager):
        self.system = system
        self.session_manager = session_manager

    def handle_verification_failure(self, student_id: str, error_type: str, details: str = "") -> Dict:
        """Handle verification failures with specific recovery options"""
        error_response = {
            "success": False,
            "student_id": student_id,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }

        # Error type specific responses
        error_messages = {
            "wifi": {
                "message": "WiFi verification failed",
                "recovery_options": [
                    "Reconnect to classroom WiFi",
                    "Verify you are in the correct classroom",
                    "Check if the session is still active",
                    "Contact teacher for manual verification"
                ]
            },
            "face": {
                "message": "Face verification failed",
                "recovery_options": [
                    "Ensure good lighting",
                    "Remove face coverings",
                    "Face the camera directly",
                    "Try again with better positioning"
                ]
            },
            "session": {
                "message": "Session verification failed",
                "recovery_options": [
                    "Verify session ID is correct",
                    "Check if session is still active",
                    "Ask teacher to verify session status"
                ]
            },
            "quality": {
                "message": "Image quality insufficient",
                "recovery_options": [
                    "Find better lighting",
                    "Ensure clear face view",
                    "Remove any obstructions",
                    "Keep still during capture"
                ]
            },
            "input": {
                "message": "Invalid input data",
                "recovery_options": [
                    "Verify all required fields",
                    "Check input format",
                    "Try again with correct information"
                ]
            },
            "capture": {
                "message": "Image capture failed",
                "recovery_options": [
                    "Check camera connection",
                    "Ensure camera permissions",
                    "Try again with better lighting",
                    "Restart the application"
                ]
            }
        }

        # Get error specific message and options or use default
        error_info = error_messages.get(error_type, {
            "message": f"Verification failed: {details}",
            "recovery_options": [
                "Try again",
                "Contact system administrator"
            ]
        })

        error_response.update(error_info)

        # Display error message and recovery options
        ProgressIndicator.show_error(f"\n{error_response['message']}")
        print("\nRecovery Options:")
        for i, option in enumerate(error_response['recovery_options'], 1):
            print(f"{i}. {option}")

        return error_response

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
            network_result = self.session_manager.verify_student_network(session_id)
            if not network_result["success"]:
                return self.handle_verification_failure(
                    student_id,
                    "wifi",
                    network_result["message"]
                )

            # Capture and verify face
            MenuHandler.display_verification_instructions()
            image_path, captured_image = capture_image()

            if image_path is None or captured_image is None:
                return self.handle_verification_failure(
                    student_id,
                    "capture",
                    "Image capture cancelled"
                )

            try:
                verification_result = self.system.verify_attendance_with_wifi(
                    student_id=student_id,
                    captured_image=captured_image,
                    session_id=session_id,
                    wifi_data=get_device_network_info(),
                    session_manager=self.session_manager
                )

                if verification_result["success"]:
                    return verification_result
                else:
                    return self.handle_verification_failure(
                        student_id,
                        verification_result.get("error_type", "unknown"),
                        verification_result.get("message", "Verification failed")
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

            # Get student ID for testing
            student_id = MenuHandler.get_user_input("Enter student ID for testing: ")
            if not student_id:
                return self.handle_verification_failure(
                    "test",
                    "input",
                    "Student ID is required"
                )

            # Create test session
            session_result = self.session_manager.start_session(
                teacher_id="TEST_TEACHER",
                hall_id="TEST_HALL"
            )

            if not session_result["success"]:
                return self.handle_verification_failure(
                    "test",
                    "session",
                    f"Failed to create test session: {session_result['message']}"
                )

            session_id = session_result["session_id"]
            MenuHandler.display_verification_instructions()
            image_path, captured_image = capture_image()

            if image_path is None or captured_image is None:
                return self.handle_verification_failure(
                    student_id,
                    "capture",
                    "Image capture cancelled"
                )

            try:
                verification_result = self.system.verify_attendance_with_wifi(
                    student_id=student_id,
                    captured_image=captured_image,
                    session_id=session_id,
                    wifi_data=network_info,
                    session_manager=self.session_manager
                )

                if verification_result["success"]:
                    return {
                        "success": True,
                        "message": "Test verification successful",
                        "data": verification_result["data"],
                        "test_details": {
                            "network_info": network_info,
                            "session_id": session_id
                        }
                    }
                else:
                    return self.handle_verification_failure(
                        student_id,
                        verification_result.get("error_type", "unknown"),
                        verification_result.get("message", "Test verification failed")
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

# ===============================
# MAIN EXECUTION
# ===============================
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
        session_manager = SessionManager()
        
        # Initialize handlers
        session_handler = SessionHandler(system, session_manager)
        verification_handler = VerificationHandler(system, session_manager)
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
                            f"\nConfidence: {result['data']['face_confidence']:.3f}" +
                            f"\nTime: {result['data']['verification_time']:.2f}s\n" +
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
                            print(f"Duration: {result['summary']['duration']}")
                    else:
                        ProgressIndicator.show_error(result["message"])
                
                elif choice == "4":
                    result = await verification_handler.handle_test_verification()
                    if result["success"]:
                        ProgressIndicator.show_success(
                            "\nTest Verification Results:\n" +
                            "-" * 40 +
                            f"\nStatus: Successful" +
                            f"\nFace Confidence: {result['data']['face_confidence']:.3f}" +
                            f"\nVerification Time: {result['data']['verification_time']:.2f}s" +
                            f"\nWiFi Verified: {result['data']['wifi_verified']}" +
                            f"\nTimestamp: {result['data']['timestamp']}"
                        )
                        
                        # Display detailed test metrics
                        if "test_details" in result:
                            print("\nTest Details:")
                            print("-" * 40)
                            print(f"Network: {result['test_details']['network_info']['ip_address']}")
                            print(f"Session ID: {result['test_details']['session_id']}")

                    else:
                        ProgressIndicator.show_error(
                            f"\nTest Verification Failed:" +
                            f"\nReason: {result['message']}"
                        )
                        
                        if "recovery_options" in result:
                            print("\nTroubleshooting Tips:")
                            for i, option in enumerate(result["recovery_options"], 1):
                                print(f"{i}. {option}")

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
                        if "details" in result:
                            print("\nRegistration Details:")
                            for key, value in result["details"].items():
                                print(f"{key}: {value}")
                        
                elif choice == "6":
                    ProgressIndicator.show_status("Cleaning up and shutting down...")
                    break
                
                else:
                    ProgressIndicator.show_warning("Invalid choice. Please select 1-6.")
                
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
           