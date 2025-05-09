from datetime import datetime
from typing import Dict
import logging
from data.structures import WifiSession

class WifiVerificationSystem:
    """Handles WiFi verification for attendance system"""
    
    def __init__(self):
        self.active_sessions = {}
        
    def create_session(self, teacher_id: str, hall_id: str, wifi_ssid: str) -> Dict:
        """Create a new teaching session"""
        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hall_id}"
            
            session = WifiSession(
                session_id=session_id,
                teacher_id=teacher_id,
                hall_id=hall_id,
                wifi_ssid=wifi_ssid,
                start_time=datetime.now()
            )
            
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
        """Verify if student is connected to the correct WiFi network"""
        try:
            if session_id not in self.active_sessions:
                return {
                    "success": False,
                    "message": "No active teaching session found"
                }
            
            session = self.active_sessions[session_id]
            
            if not session.is_active:
                return {
                    "success": False,
                    "message": "Teaching session has ended"
                }
            
            if student_wifi_data.get("ssid") != session.wifi_ssid:
                return {
                    "success": False,
                    "message": "Not connected to the correct WiFi network"
                }
            
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