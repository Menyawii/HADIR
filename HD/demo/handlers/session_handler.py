from typing import Dict
from demo.ui.progress_indicator import ProgressIndicator
from demo.handlers.menu_handler import MenuHandler

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
        
        # Start attendance session
        session_result = self.attendance_session.start_session(
            teacher_id=teacher_id,
            hall_id=hall_id
        )
        
        if session_result["success"]:
            # Create WiFi session
            wifi_result = self.wifi_system.create_session(
                teacher_id=teacher_id,
                hall_id=hall_id,
                wifi_ssid=self.attendance_session._get_device_ip()
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