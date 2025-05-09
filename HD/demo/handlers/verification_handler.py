from datetime import datetime
import logging
import os
from typing import Dict

from demo.handlers.menu_handler import MenuHandler
from demo.handlers.camera_handler import capture_image
from demo.ui.progress_indicator import ProgressIndicator
from core.utils.config import Config
from core.models.image_processor import ImagePreprocessor

class VerificationHandler:
    def __init__(self, system, attendance_session, wifi_system):
        self.system = system
        self.attendance_session = attendance_session
        self.wifi_system = wifi_system

    def handle_verification_failure(self, student_id: str, error_type: str, details: str = "") -> Dict:
        """Handle verification failures with recovery options"""
        error_response = {
            "success": False,
            "student_id": student_id,
            "error_type": error_type,
            "message": details,  # Add the message key
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "recovery_options": self._get_recovery_options(error_type)
        }

        # Display error message and recovery options
        ProgressIndicator.show_error(f"\n{error_response['message']}")
        print("\nRecovery Options:")
        for i, option in enumerate(error_response['recovery_options'], 1):
            print(f"{i}. {option}")

        return error_response
    def _get_recovery_options(self, error_type: str) -> list:
        """Get recovery options based on error type"""
        options = {
            "wifi": [
                "Reconnect to classroom WiFi",
                "Verify you are in the correct classroom",
                "Check if the session is still active",
                "Contact teacher for manual verification"
            ],
            "face": [
                "Ensure good lighting",
                "Remove face coverings",
                "Face the camera directly",
                "Try again with better positioning"
            ],
            "session": [
                "Verify session ID is correct",
                "Check if session is still active",
                "Ask teacher to verify session status"
            ],
            "quality": [
                "Find better lighting",
                "Ensure clear face view",
                "Remove any obstructions",
                "Keep still during capture"
            ],
            "input": [
                "Verify all required fields",
                "Check input format",
                "Try again with correct information"
            ],
            "capture": [
                "Check camera connection",
                "Ensure camera permissions",
                "Try again with better lighting",
                "Restart the application"
            ]
        }
        return options.get(error_type, ["Try again", "Contact system administrator"])

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
                    "network",
                    network_result["message"]
                )

            # Verify student exists
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
                # Verify student
                result = self.system.verify_student(student_id, img)
                
                if result.success:
                    # Mark attendance
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
                    
                    return {
                        "success": True,
                        "message": "Verification successful",
                        "data": {
                            "student_id": student_id,
                            "session_id": session_id,
                            "confidence_score": result.confidence_score,
                            "verification_time": result.verification_time,
                            "verification_type": result.verification_type,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                
                error_message = result.error_message or "Verification failed"
                return self.handle_verification_failure(
                    student_id,
                    "face",
                    error_message
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