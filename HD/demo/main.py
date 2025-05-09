import asyncio
import logging
import os
from typing import Dict

from core.utils.config import Config
from core.utils.exceptions import SystemInitializationError, FaceRecognitionError, CameraError
from core.models.face_recognition import FaceRecognitionSystem
from core.session.attendance import AttendanceSession
from core.session.wifi_verification import WifiVerificationSystem

from demo.ui.progress_indicator import ProgressIndicator
from demo.ui.network_display import NetworkInfoDisplay
from demo.handlers.menu_handler import MenuHandler
from demo.handlers.session_handler import SessionHandler
from demo.handlers.verification_handler import VerificationHandler
from demo.handlers.registration_handler import StudentRegistrationHandler



class DemoApplication:
    @staticmethod
    async def run():
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
                        # Start New Session
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
                        # Verify Attendance
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
                        # End Session
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
                        # Test Verification
                        result = await verification_handler.handle_verification()
                        if result["success"]:
                            ProgressIndicator.show_success(
                                "\nTest Verification Results:\n" +
                                "-" * 40 +
                                f"\nStatus: Successful" +
                                f"\nFace Confidence: {result['data']['confidence_score']:.3f}" +
                                f"\nTimestamp: {result['data']['timestamp']}"
                            )
                        else:
                            ProgressIndicator.show_error(
                                f"\nTest Verification Failed:" +
                                f"\nReason: {result['message']}"
                            )
                            
                            # Show troubleshooting tips
                            if "recovery_options" in result:
                                print("\nTroubleshooting Tips:")
                                for i, option in enumerate(result["recovery_options"], 1):
                                    print(f"{i}. {option}")

                    elif choice == "5":
                        # Register New Student
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
                            
                    elif choice == "6":
                        # Exit
                        ProgressIndicator.show_status("Cleaning up and shutting down...")
                        break
                    
                    else:
                        ProgressIndicator.show_warning("Invalid choice. Please select 1-6.")
                    
                    # Add a small delay between operations
                    await asyncio.sleep(1)
                        
                except Exception as e:
                    ProgressIndicator.show_error(f"Operation error: {str(e)}")
                    logging.error(f"Operation error: {str(e)}")
                    await asyncio.sleep(2)
                    
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
        raise Exception(f"Directory setup failed: {str(e)}")

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

# Initialize logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=Config.LOG_FILE
)

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

