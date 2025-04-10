from deepface import DeepFace 
import cv2
import os
import time

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        filename = f"captured_image_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
    cap.release()
    cv2.destroyAllWindows()
    return filename

def generate_encoding(image_path):
    try:
        encoding = DeepFace.represent(img_path=image_path, model_name="Facenet")
        return encoding
    except Exception as e:
        print(f"Error generating encoding for {image_path}: {e}")
        return None

def verify_face(new_encoding):
    if new_encoding is None:
        return "No face detected."

    matches = []

    for filename in os.listdir():
        if filename.endswith(".jpg") and not filename.startswith("captured_image"):
            try:
                # Generate encoding for the stored image
                stored_encoding = generate_encoding(filename)
                
                if stored_encoding is not None:
                    # Use DeepFace.verify with image paths
                    result = DeepFace.verify(img1_path=new_encoding[0]['embedding'], img2_path=stored_encoding[0]['embedding'], model_name="Facenet")
                    if result['verified']:
                        matches.append(filename)
            except Exception as e:
                print(f"Error verifying with {filename}: {e}")

    if matches:
        return f"Matches found with: {', '.join(matches)}."
    else:
        return "No match found."
    
def main():
    # Capture a new image
    image_filename = capture_image()
    
    # Generate encoding for the captured image
    new_encoding = generate_encoding(image_filename)
   
    # Verify the encoding against stored images
    result = verify_face(new_encoding)
    print(result)

if __name__ == "__main__":
    main()