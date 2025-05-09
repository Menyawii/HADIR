import os
import pickle
import logging
from typing import Dict, Optional, List
from deepface import DeepFace

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
        """Get or generate face encoding for an image"""
        if image_path in self.cache:
            return self.cache[image_path]
        
        try:
            # Generate new encoding
            encoding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet"
            )
            
            if encoding:
                self.cache[image_path] = encoding
                self.save_cache()
                return encoding
                
        except Exception as e:
            logging.error(f"Encoding generation error: {str(e)}")
        
        return None