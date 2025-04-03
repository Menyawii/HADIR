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

def preprocess_image(filename):
    frame = cv2.imread(filename)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face on the original image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        processed_filename = f"processed_{filename}"
        cv2.imwrite(processed_filename, frame)  # Save the original frame
        return processed_filename

    return None

def generate_encoding(image_path):
    try:
        encoding = DeepFace.represent(img_path=image_path, model_name="Facenet")
        return encoding
    except Exception as e:
        print(f"Error generating encoding for {image_path}: {e}")
        return None

def verify_face(processed_image_path):
    if processed_image_path is None:
        return "No face detected."

    matches = []

    for filename in os.listdir():
        if filename.endswith(".jpg") and not filename.startswith("captured_image"):
            try:
                # Use DeepFace.verify with image paths
                result = DeepFace.verify(img1_path=processed_image_path, img2_path=filename, model_name="Facenet")
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

    # Preprocess the captured image
    processed_filename = preprocess_image(image_filename)

    # Generate encoding for the processed image
    new_encoding = generate_encoding(processed_filename)

    # Verify the encoding against stored images
    result = verify_face(new_encoding)
    print(result)

if __name__ == "__main__":
    main()
