from datetime import datetime
from typing import Dict, Set
import logging
from data.structures import Session, SessionStatus

class AttendanceSession:
    """Enhanced attendance session management"""
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}

    def start_session(self, teacher_id: str, hall_id: str) -> Dict:
        """Start new teaching session with enhanced tracking"""
        try:
            # Generate session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hall_id}"

            # Create new session
            self.active_sessions[session_id] = Session(
                teacher_id=teacher_id,
                hall_id=hall_id,
                start_time=datetime.now().isoformat(),
                teacher_ip=self._get_device_ip(),
                is_active=True
            )

            logging.info(f"Session created: {session_id}")

            return {
                "success": True,
                "message": "Session started successfully",
                "session_id": session_id
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

            # Get student's network info
            student_ip = self._get_device_ip()
            
            # Basic subnet comparison
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
        """End a teaching session with attendance summary"""
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
            session.status = SessionStatus.COMPLETED
            session.end_time = datetime.now().isoformat()
            
            # Generate session summary
            summary = {
                "total_students": len(session.attendance_records),
                "connected_devices": len(session.connected_students),
                "start_time": session.start_time,
                "end_time": session.end_time
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

    def _get_device_ip(self) -> str:
        """Get device IP address"""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logging.error(f"Error getting IP: {str(e)}")
            return "unknown"