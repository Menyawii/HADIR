from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from datetime import datetime
from enum import Enum

@dataclass
class RecognitionResult:
    """Enhanced recognition result structure"""
    def __init__(
        self,
        success: bool,
        error_message: Optional[str] = None,
        confidence_score: Optional[float] = None,
        verification_time: Optional[float] = None,
        verification_type: Optional[str] = None,
        quality_details: Optional[Dict] = None,
        data: Optional[Dict] = None
    ):
        self.success = success
        self.error_message = error_message
        self.confidence_score = confidence_score
        self.verification_time = verification_time
        self.verification_type = verification_type
        self.quality_details = quality_details or {}
        self.data = data or {}

    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "confidence_score": self.confidence_score,
            "verification_time": self.verification_time,
            "verification_type": self.verification_type,
            "quality_details": self.quality_details,
            "data": self.data
        }

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