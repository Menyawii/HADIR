import os
import time
from datetime import datetime
from typing import Dict
import logging

from core.utils.config import Config  # Import Config directly
from demo.handlers.menu_handler import MenuHandler
from demo.handlers.camera_handler import capture_image
from core.models.face_recognition import FaceRecognitionSystem
from core.models.image_processor import ImagePreprocessor

class StudentRegistrationHandler:
    def __init__(self, face_recognition_system: FaceRecognitionSystem):
        self.system = face_recognition_system

    async def handle_registration(self) -> Dict:
        """Handle new student registration process"""
        try:
            print("\n=== New Student Registration ===")
            
            # Display instructions and capture face image
            MenuHandler.display_registration_instructions()
            image_path, captured_image = capture_image()

            if image_path is None or captured_image is None:
                return {
                    "success": False,
                    "message": "Image capture cancelled"
                }

            try:
                # Check image quality
                quality_result = ImagePreprocessor.check_face_quality(captured_image)
                if not quality_result:
                    return {
                        "success": False,
                        "message": "Image quality not sufficient for registration"
                    }

                # Get student information
                student_info = self._get_student_info()
                if not student_info["success"]:
                    return student_info

                # Generate unique student ID
                student_id = f"STU_{int(time.time())}"

                # Generate face encoding
                encoding_result = self.system.get_face_encoding_for_storage(captured_image)
                
                if not encoding_result["success"]:
                    return {
                        "success": False,
                        "message": f"Failed to generate face encoding: {encoding_result['message']}"
                    }

                # Save student image - using Config directly
                student_image_path = os.path.join(
                    Config.STORED_IMAGES_DIR,  # Use Config directly instead of self.system.Config
                    f"{student_id}.jpg"
                )
                os.makedirs(os.path.dirname(student_image_path), exist_ok=True)
                os.rename(image_path, student_image_path)

                return {
                    "success": True,
                    "message": "Student registered successfully",
                    "student_id": student_id,
                    "details": {
                        "email": student_info["email"],
                        "registration_time": datetime.now().isoformat()
                    }
                }

            finally:
                # Cleanup temporary image
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