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
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

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
    """Stores face recognition results"""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    verification_time: Optional[float] = None

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
    name: str
    student_id: str
    course: str
    department: str
    role: UserRole = UserRole.STUDENT
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
    id: str
    course_id: str
    teacher_id: str
    hall_id: str
    start_time: str
    end_time: str
    wifi_ssid: str
    rssi_threshold: float
    status: SessionStatus
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: Optional[str] = None

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
# SYSTEM INTERFACES
# ===============================
class IAuthenticationSystem:
    """Authentication system interface - Backend team to implement"""
    def register_student(self, student_data: Dict, face_image: np.ndarray) -> Dict:
        raise NotImplementedError

    def login(self, email: str, password: str) -> Dict:
        raise NotImplementedError

    def verify_session(self, session_token: str) -> Dict:
        raise NotImplementedError

class IAttendanceSystem:
    """Attendance system interface - Backend team to implement"""
    def start_session(self, teacher_id: str, course_id: str, hall_id: str) -> Dict:
        raise NotImplementedError

    def verify_attendance(self, student_id: str, session_id: str, 
                         face_image: np.ndarray, wifi_data: Dict) -> Dict:
        raise NotImplementedError

    def manual_attendance(self, teacher_id: str, student_id: str, 
                         session_id: str, status: AttendanceStatus, reason: str) -> Dict:
        raise NotImplementedError

    def get_attendance_records(self, user_id: str, role: UserRole, 
                             filters: Dict = None) -> List[Dict]:
        raise NotImplementedError

class INotificationSystem:
    """Notification system interface - Backend team to implement"""
    def send_attendance_confirmation(self, student_id: str) -> bool:
        raise NotImplementedError

    def send_verification_failure(self, student_id: str, teacher_id: str, 
                                reason: str) -> bool:
        raise NotImplementedError

    def send_session_notification(self, session_id: str, notification_type: str) -> bool:
        raise NotImplementedError

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

def capture_image() -> Optional[str]:
    """Capture image from webcam with preview"""
    try:
        with camera_context() as cap:
            if not cap.isOpened():
                raise CameraError("Failed to open webcam")

            ProgressIndicator.show_status("Camera ready. Position your face in the frame...")
            
            # Create window for preview
            cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise CameraError("Failed to capture frame")

                # Show preview with guidelines
                preview_frame = frame.copy()
                height, width = preview_frame.shape[:2]
                
                # Draw center guidelines
                center_x = width // 2
                center_y = height // 2
                size = min(width, height) // 3
                
                # Draw rectangle guide
                cv2.rectangle(preview_frame, 
                            (center_x - size, center_y - size),
                            (center_x + size, center_y + size),
                            (0, 255, 0), 2)

                # Add instructions
                cv2.putText(preview_frame, "Position face within the green box", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview_frame, "Press SPACE to capture or Q to quit", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show preview
                cv2.imshow('Camera Preview', preview_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return None
                elif key == ord(' '):  # Space bar
                    # Capture the region of interest
                    roi = frame[center_y-size:center_y+size, center_x-size:center_x+size]
                    if roi.size > 0:
                        break
                    else:
                        ProgressIndicator.show_warning("Invalid capture region. Please try again.")
                        continue

            cv2.destroyAllWindows()
            
            # Save the captured image
            filename = f"captured_image_{int(time.time())}.jpg"
            cv2.imwrite(filename, roi)
            
            ProgressIndicator.show_success(f"Image captured successfully")
            return filename

    except Exception as e:
        ProgressIndicator.show_error(f"Capture error: {str(e)}")
        logging.error(f"Capture error: {str(e)}")
        return None
    finally:
        cv2.destroyAllWindows()

def check_image_quality(image: np.ndarray) -> bool:
    """Check if image meets quality requirements"""
    try:
        # Check image size
        if image.shape[0] < 100 or image.shape[1] < 100:
            return False
            
        # Check brightness
        brightness = np.mean(image)
        if brightness < 40 or brightness > 250:
            return False
            
        # Check contrast
        contrast = np.std(image)
        if contrast < 20:
            return False
            
        return True
    except:
        return False
    
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

    def verify_student(self, student_id: str, captured_image: np.ndarray) -> RecognitionResult:
        """Verifies student identity using facial recognition"""
        try:
            # Get stored encoding
            stored_encoding = self.get_student_encoding(student_id)
            if stored_encoding is None:
                return RecognitionResult(
                    success=False,
                    error_message="No stored encoding found"
                )

            # Generate new encoding
            captured_encoding = self.generate_live_encoding(captured_image)
            if captured_encoding is None:
                return RecognitionResult(
                    success=False,
                    error_message="Failed to generate encoding"
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
                verification_time=verification_time
            )

        except Exception as e:
            logging.error(f"Verification error: {str(e)}")
            return RecognitionResult(
                success=False,
                error_message=str(e)
            )

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
        try:
            preprocessed = self.image_preprocessor.preprocess_image(image)
            if preprocessed is None:
                return None
                
            temp_path = os.path.join(Config.TEMP_IMAGE_DIR, f"temp_{int(time.time())}.jpg")
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))
            
            encoding = generate_encoding(temp_path)
            os.remove(temp_path)
            return encoding
            
        except Exception as e:
            logging.error(f"Error generating live encoding: {str(e)}")
            return None

    def register_new_student(self, student_id: str, image: np.ndarray) -> Dict:
        """Registers new student with facial encoding"""
        try:
            # Basic image checks
            if image is None or image.size == 0:
                return {
                    "success": False,
                    "message": "Invalid image data"
                }

            # Resize image first
            resized_image = cv2.resize(image, Config.IMAGE_SIZE)

            # Basic preprocessing without face detection
            preprocessed_image = cv2.convertScaleAbs(resized_image, alpha=1.3, beta=5)

            # Save the preprocessed image
            save_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
            cv2.imwrite(save_path, preprocessed_image)

            # Generate and cache encoding
            encoding = self.encoding_cache.get_encoding(save_path)
            if encoding is None:
                os.remove(save_path)  # Clean up if encoding fails
                return {
                    "success": False,
                    "message": "Failed to generate face encoding"
                }

            # Update stored images list
            if save_path not in self.stored_images:
                self.stored_images.append(save_path)

            return {
                "success": True,
                "message": "Student registered successfully"
            }

        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}"
            }

    def process_verification(self, captured_image_path: str) -> Dict:
        """Processes general verification against all stored images"""
        try:
            new_encoding = self.encoding_cache.get_encoding(captured_image_path)
            if new_encoding is None:
                raise FaceRecognitionError("Failed to generate encoding")

            matches = []
            confidence_scores = {}

            for stored_image in self.stored_images:
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

            return {
                "status": "success",
                "matches": matches,
                "confidence_scores": confidence_scores
            }

        except Exception as e:
            logging.error(f"Verification error: {str(e)}")
            return {"status": "error", "message": str(e)}

# ===============================
# MAIN EXECUTION
# ===============================
async def main():
    """Main execution function with enhanced user interface and error handling"""
    try:
        # Initial setup and checks
        ProgressIndicator.show_status("Checking system requirements...")
        if not check_requirements():
            ProgressIndicator.show_error("System requirements not met. Please check dependencies.")
            sys.exit(1)

        ProgressIndicator.show_status("Setting up system directories...")
        setup_directories()
        
        ProgressIndicator.show_status("Initializing Face Recognition System...")
        system = FaceRecognitionSystem()
        
        while True:
            print("\n" + "="*40)
            print("Face Recognition Attendance System")
            print("="*40)
            print("1. Register New Student")
            print("2. Verify Attendance")
            print("3. Test General Verification")
            print("4. Exit")
            print("-"*40)
            
            choice = input("Select an option (1-4): ")

            if choice == "1":
                # Register new student
                print("\n=== Student Registration ===")
                student_id = input("Enter student ID: ").strip()
                
                if not student_id:
                    ProgressIndicator.show_error("Student ID cannot be empty")
                    continue
                
                # Check if student already exists
                if os.path.exists(os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")):
                    ProgressIndicator.show_warning("Student ID already exists")
                    continue_reg = input("Do you want to overwrite? (y/n): ").lower()
                    if continue_reg != 'y':
                        continue

                ProgressIndicator.show_status("\nPreparing to capture registration image...")
                ProgressIndicator.show_status("Please ensure:")
                print("- Good lighting on your face")
                print("- Look directly at the camera")
                print("- Keep a neutral expression")
                print("- Maintain 2-3 feet distance")
                print("\nPress SPACE to capture or Q to cancel")
                
                image_path = capture_image()
                
                if image_path is None:
                    ProgressIndicator.show_warning("Registration cancelled")
                    continue

                img = cv2.imread(image_path)
                if img is None:
                    ProgressIndicator.show_error("Failed to read captured image")
                    continue

                # Check image quality
                if not ImagePreprocessor.check_face_quality(img):
                    ProgressIndicator.show_error(
                        "Image quality too low. Please ensure good lighting and clear view"
                    )
                    continue

                result = system.register_new_student(student_id, img)
                
                if result["success"]:
                    ProgressIndicator.show_success(result["message"])
                    print("\nRegistration Details:")
                    print(f"Student ID: {student_id}")
                    print(f"Image stored as: {student_id}.jpg")
                else:
                    ProgressIndicator.show_error(result["message"])

            elif choice == "2":
                # Verify attendance
                print("\n=== Attendance Verification ===")
                student_id = input("Enter student ID: ").strip()
                
                if not student_id:
                    ProgressIndicator.show_error("Student ID cannot be empty")
                    continue
                
                # Check if student exists
                if not os.path.exists(os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")):
                    ProgressIndicator.show_error("Student ID not found in system")
                    continue

                ProgressIndicator.show_status("\nPreparing to capture verification image...")
                ProgressIndicator.show_status("Please maintain the same position as registration")
                print("\nPress SPACE to capture or Q to cancel")
                
                image_path = capture_image()
                
                if image_path is None:
                    ProgressIndicator.show_warning("Verification cancelled")
                    continue

                img = cv2.imread(image_path)
                if img is None:
                    ProgressIndicator.show_error("Failed to read captured image")
                    continue

                result = system.verify_student(student_id, img)
                
                if result.success:
                    ProgressIndicator.show_success(
                        f"Verification successful!\n"
                        f"Confidence Score: {result.confidence_score:.3f}\n"
                        f"Verification Time: {result.verification_time:.2f} seconds"
                    )
                    # Add timestamp
                    print(f"\nAttendance marked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    ProgressIndicator.show_error(
                        f"Verification failed: {result.error_message}"
                    )

            elif choice == "3":
                # General verification
                print("\n=== General Verification Test ===")
                ProgressIndicator.show_status("This will compare your face against all stored images")
                print("\nPress SPACE to capture or Q to cancel")
                
                image_path = capture_image()
                if image_path is None:
                    ProgressIndicator.show_warning("Verification cancelled")
                    continue

                ProgressIndicator.show_status("Processing verification...")
                result = system.process_verification(image_path)
                
                if result["status"] == "success":
                    if result["matches"]:
                        ProgressIndicator.show_success(f"Found {len(result['matches'])} matches!")
                        print("\nMatch Details:")
                        print("-" * 40)
                        for match in result["matches"]:
                            confidence = result["confidence_scores"][match]
                            student_id = os.path.basename(match).replace('.jpg', '')
                            print(f"Student ID: {student_id}")
                            print(f"Confidence: {confidence:.3f}")
                            print("-" * 40)
                    else:
                        ProgressIndicator.show_warning("No matches found in database")
                else:
                    ProgressIndicator.show_error(f"Error: {result['message']}")

            elif choice == "4":
                ProgressIndicator.show_status("Cleaning up and shutting down...")
                break

            else:
                ProgressIndicator.show_warning("Invalid choice. Please select 1-4.")

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
            logging.error(f"Cleanup error during shutdown: {str(e)}")

if __name__ == "__main__":
    # Set up basic configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    asyncio.run(main())