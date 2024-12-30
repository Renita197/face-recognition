import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time
import pyttsx3
import torch
import cv2
import numpy as np
from PIL import Image

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "<sender_email>"
SENDER_PASSWORD = "<sender_password>"  
RECEIVER_EMAIL = "<receiver_email>"  

def train_face_model(dataset_path):
    # Train the face recognition model with multiple images per person
    known_face_encodings = []
    known_face_names = []
    
    print("Training face recognition model...")
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            person_encodings = []
            successful_encodings = 0
            
            # Process each image for the person
            for image_name in os.listdir(person_folder):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_folder, image_name)
                    try:
                        # Load and encode face with different detection models
                        image = face_recognition.load_image_file(image_path)
                        
                        face_locations = face_recognition.face_locations(image, model="cnn")
                        if not face_locations:
                            face_locations = face_recognition.face_locations(image, model="hog")
                        
                        if face_locations:
                            # Get multiple encodings if multiple faces are detected
                            encodings = face_recognition.face_encodings(image, face_locations, num_jitters=10)
                            for encoding in encodings:
                                person_encodings.append(encoding)
                                successful_encodings += 1
                            print(f"Successfully processed {image_name} for {person_name} - Found {len(encodings)} faces")
                        else:
                            print(f"No face found in {image_name}")
                    except Exception as e:
                        print(f"Error processing {image_name}: {e}")
            
            # Add multiple encodings for better angle coverage
            if person_encodings:
                print(f"Successfully encoded {successful_encodings} faces for {person_name}")
                for encoding in person_encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(person_name)
                print(f"Added {person_name} to recognition database with {successful_encodings} variations")
    
    return known_face_encodings, known_face_names

def get_face_distance(face_encoding, known_face_encodings):
    """Calculate face distances and return the closest match with confidence"""
    if len(known_face_encodings) == 0:
        return None, None, 0
    
    # Calculate distances to all known faces
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    
    # Get all matches for this person
    matches = []
    for i, distance in enumerate(face_distances):
        if distance < 0.6:  # Increased threshold for better unknown face detection
            matches.append((i, distance))
    
    if matches:
        # Sort by distance and get the best match
        matches.sort(key=lambda x: x[1])
        best_match_index, min_distance = matches[0]
        
        # Calculate confidence score (0-100%)
        confidence = (1 - min_distance) * 100
        
        return best_match_index, min_distance, confidence
    
    return None, None, 0

def send_alert_email(frame, timestamp):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = 'Security Alert: Unknown Person Detected!'
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        # Add text
        text = f"An unknown person was detected at your home at {timestamp}"
        msg.attach(MIMEText(text))

        # Save and attach the image
        image_path = f"unknown_person_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(image_path, frame)
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
        msg.attach(img)

        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        # Clean up
        os.remove(image_path)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def speak_greeting(name):
    """Speak a greeting for the recognized person"""
    try:
        greeting = f"Hello {name}, Welcome home!"
        engine.say(greeting)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in speech: {e}")

def load_model():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s-face.pt')
    model.conf = 0.5  # Confidence threshold
    model.iou = 0.45  # NMS IOU threshold
    return model

def detect_faces(image_path):
    # Load the model
    model = load_model()
    
    # Read image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform detection
    results = model(img_rgb)
    
    # Process results
    detections = results.pandas().xyxy[0].values
    
    # Draw bounding boxes
    for detection in detections:
        x1, y1, x2, y2, conf, cls, label = detection
        if conf > 0.5:  # Confidence threshold
            # Draw rectangle
            cv2.rectangle(img, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 2)
            
            # Add label
            label = f'Face: {conf:.2f}'
            cv2.putText(img, label, 
                       (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
    
    return img

def process_video(video_path=0):
    # Load the model
    model = load_model()
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        
        # Process results
        detections = results.pandas().xyxy[0].values
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2, conf, cls, label = detection
            if conf > 0.5:
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                label = f'Face: {conf:.2f}'
                cv2.putText(frame, label, 
                           (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Face Detection', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Define the path to your dataset folder
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "<your_dataset_path>")
    
    print("Starting face recognition system...")
    known_face_encodings, known_face_names = train_face_model(dataset_path)

    if not known_face_encodings:
        print("Error: No faces were trained. Please check your dataset directory.")
        exit()

    print(f"Model trained with {len(known_face_names)} people: {', '.join(known_face_names)}")

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        exit()

    # Process video feed
    print("Starting video capture... Press 'q' to quit")
    last_email_time = time.time() - 300  # Initialize last email time
    email_cooldown = 300  # 5 minutes cooldown between emails

    # Track last greetings to avoid repetition
    last_greetings = {}
    greeting_cooldown = 30 

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Try CNN model first for better accuracy
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        if not face_locations:
            # Fallback to HOG model if CNN doesn't detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        if face_locations:
            # Use more jitters for better accuracy
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=3)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Get best match and confidence
                best_match_index, min_distance, confidence = get_face_distance(face_encoding, known_face_encodings)
                
                if best_match_index is not None and confidence > 50:  # Lower threshold for better detection
                    name = known_face_names[best_match_index]
                    confidence_text = f"{confidence:.1f}%"
                    color = (0, 255, 0)  # Green for known face
                    
                    # Check if enough time has passed since last greeting for this person
                    current_time = time.time()
                    if name not in last_greetings or (current_time - last_greetings[name]) >= greeting_cooldown:
                        speak_greeting(name)

                       
                    name = "Unknown"
                    confidence_text = ""
                    color = (0, 0, 255)  # Red for unknown face
                    
                    # Send email alert if cooldown period has passed
                    current_time = time.time()
                    if current_time - last_email_time >= email_cooldown:
                        timestamp = datetime.now()
                        if send_alert_email(frame, timestamp):
                            print(f"Alert email sent for unknown person at {timestamp}")
                            last_email_time = current_time
                
                # Draw box and label with confidence
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                label = f"{name} {confidence_text}"
                cv2.putText(frame, label, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Display the video feed
        cv2.imshow("Face Recognition", frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    print("Face recognition system stopped")

if __name__ == "__main__":
    main()
