import os
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging

from core.utils.config import Config
from demo.handlers.menu_handler import MenuHandler
from demo.handlers.camera_handler import capture_image
from core.models.face_recognition import FaceRecognitionSystem
from core.models.image_processor import ImagePreprocessor
from demo.ui.progress_indicator import ProgressIndicator

class StudentRegistrationHandler:
    def __init__(self, face_recognition_system: FaceRecognitionSystem):
        self.system = face_recognition_system

    async def handle_registration(self) -> Dict:
        """Handle new student registration with face duplication checking"""
        try:
            print("\n=== New Student Registration ===")
            
            # Display instructions and capture face image FIRST (before asking for ID)
            MenuHandler.display_registration_instructions()
            image_path, captured_image = capture_image()

            if image_path is None or captured_image is None:
                return {
                    "success": False,
                    "message": "Image capture cancelled"
                }
                
            # Check image quality
            quality_result = ImagePreprocessor.check_face_quality(captured_image)
            if not quality_result:
                return {
                    "success": False,
                    "message": "Image quality not sufficient for registration"
                }
                
            # CHECK FOR FACE DUPLICATES - Check if this face already exists under any ID
            duplicate_check = self.check_face_duplicates(captured_image)
            if duplicate_check["duplicate_found"]:
                ProgressIndicator.show_warning(
                    f"This face appears to be already registered as: {duplicate_check['student_id']}"
                )
                
                # Show the matching face
                view_option = MenuHandler.get_user_input("View existing registration? (y/n): ")
                if view_option.lower() == 'y':
                    if duplicate_check["image"] is not None:
                        cv2.namedWindow('Existing Registration', cv2.WINDOW_NORMAL)
                        cv2.imshow('Existing Registration', duplicate_check["image"])
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                
                # Ask whether to continue
                continue_option = MenuHandler.get_user_input("Continue with new registration anyway? (y/n): ")
                if continue_option.lower() != 'y':
                    return {
                        "success": False,
                        "message": "Registration cancelled - face already registered"
                    }
                
                ProgressIndicator.show_status("Continuing with registration despite face similarity...")
            
            # NOW ask for student ID (after face capture and duplicate check)
            student_id = MenuHandler.get_user_input("Enter Student ID: ")
            if not student_id:
                return {
                    "success": False,
                    "message": "Student ID is required"
                }
                
            # Check if student ID already exists
            if self.student_exists(student_id):
                ProgressIndicator.show_warning(f"Student with ID {student_id} is already registered!")
                
                # Option to view existing registration
                view_option = MenuHandler.get_user_input("View existing registration? (y/n): ")
                if view_option.lower() == 'y':
                    stored_img_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
                    stored_img = cv2.imread(stored_img_path)
                    if stored_img is not None:
                        cv2.namedWindow('Existing Registration', cv2.WINDOW_NORMAL)
                        cv2.imshow('Existing Registration', stored_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                
                # Option to overwrite
                overwrite = MenuHandler.get_user_input("Overwrite existing registration? (y/n): ")
                if overwrite.lower() != 'y':
                    return {
                        "success": False,
                        "message": "Registration cancelled - student ID already exists"
                    }
                
                ProgressIndicator.show_status("Continuing with registration (will overwrite existing data)...")

            try:
                # Get student information
                student_info = self._get_student_info()
                if not student_info["success"]:
                    return student_info

                # Generate face encoding
                encoding_result = self.system.get_face_encoding_for_storage(captured_image)
                
                if not encoding_result["success"]:
                    return {
                        "success": False,
                        "message": f"Failed to generate face encoding: {encoding_result['message']}"
                    }

                # Save student image
                student_image_path = os.path.join(
                    Config.STORED_IMAGES_DIR,
                    f"{student_id}.jpg"
                )
                os.makedirs(os.path.dirname(student_image_path), exist_ok=True)
                
                # Remove existing image if overwriting
                if os.path.exists(student_image_path):
                    os.remove(student_image_path)
                    
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
                if image_path and os.path.exists(image_path) and os.path.basename(image_path) != f"{student_id}.jpg":
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

    def check_face_duplicates(self, captured_image: np.ndarray) -> Dict:
        """Check if this face already exists in the system under any ID"""
        try:
            # Default response
            result = {
                "duplicate_found": False,
                "student_id": None,
                "confidence": 0.0,
                "image": None
            }
            
            # Get list of registered students
            students = self.list_registered_students()
            if not students:
                return result  # No students to compare against
                
            ProgressIndicator.show_status("Checking for duplicate registrations...")
            
            # Compare with each registered student
            highest_similarity = 0.0
            highest_similarity_id = None
            highest_similarity_img = None
            
            for student_id in students:
                try:
                    # Load student image
                    img_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
                    stored_img = cv2.imread(img_path)
                    
                    if stored_img is None:
                        continue
                    
                    # Use DeepFace comparison
                    # We're using the system's verification functionality here
                    verification_result = self.system.verify_student_images(stored_img, captured_image)
                    
                    if (verification_result.get("success", False) or 
                        verification_result.get("confidence_score", 0) > Config.FACE_RECOGNITION_THRESHOLD):
                        # Immediate match found
                        return {
                            "duplicate_found": True,
                            "student_id": student_id,
                            "confidence": verification_result.get("confidence_score", 0),
                            "image": stored_img
                        }
                    
                    # Track highest similarity for borderline cases
                    similarity = verification_result.get("confidence_score", 0)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        highest_similarity_id = student_id
                        highest_similarity_img = stored_img
                        
                except Exception as compare_error:
                    logging.warning(f"Error comparing with student {student_id}: {str(compare_error)}")
                    continue
            
            # Check if highest similarity is close to threshold (borderline case)
            if highest_similarity > Config.FACE_RECOGNITION_THRESHOLD * 0.8:  # 80% of threshold
                result = {
                    "duplicate_found": True,
                    "student_id": highest_similarity_id,
                    "confidence": highest_similarity,
                    "image": highest_similarity_img
                }
                
            return result
            
        except Exception as e:
            logging.error(f"Error checking face duplicates: {str(e)}")
            # Return false if there's an error
            return {
                "duplicate_found": False,
                "student_id": None,
                "confidence": 0.0,
                "image": None
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
    
    def student_exists(self, student_id: str) -> bool:
        """Check if a student with the given ID is already registered"""
        student_image_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
        return os.path.exists(student_image_path)
    
    def list_registered_students(self) -> List[str]:
        """Get a list of all registered student IDs"""
        students = []
        if os.path.exists(Config.STORED_IMAGES_DIR):
            for filename in os.listdir(Config.STORED_IMAGES_DIR):
                if filename.endswith('.jpg'):
                    student_id = filename.replace('.jpg', '')
                    students.append(student_id)
        return students