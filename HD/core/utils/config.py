import os
from dataclasses import dataclass
from typing import Tuple

class Config:
    """System configuration settings"""
    # Face Recognition Settings
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_RECOGNITION_THRESHOLD = 0.4
    IMAGE_SIZE = (224, 224)
    
    # Image Storage Paths
    TEMP_IMAGE_DIR = "temp_images/"
    STORED_IMAGES_DIR = "stored_images/"
    
    # Logging Settings
    LOG_FILE = "facial_recognition.log"
    LOG_LEVEL = "INFO" 
