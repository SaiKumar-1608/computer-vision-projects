import cv2
import numpy as np

# Load video
video_path = r"Vehicle-Detection-Classification-and-Counting\Videos\video.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize HOG descriptor with pre-trained SVM for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Custom classifier function (improved for better truck classification)
def classify_vehicle(h, w):
    aspect_ratio = w / h
    area = w * h
    
    if 2000 <= area < 26000:
        return "Car"
    else:
        return "Truck"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect vehicles using HOG+SVM
    boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    # Draw bounding boxes and classify vehicles
    for (x, y, w, h) in boxes:
        vehicle_type = classify_vehicle(h, w)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Vehicle Detection and Classification", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()