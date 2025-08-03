import cv2
import numpy as np

# Load video
video_path = r"Vehicle-Detection-Classification-and-Counting\Videos\video.mp4"
cap = cv2.VideoCapture(video_path)

# Background subtractor (to remove stationary objects)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

# Vehicle Classification based on contour area
def classify_vehicle(contour):
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    
    # Based on the area of the detected contour, classify the vehicle type
    if area < 1000:
        return "Car"
    elif 1000 <= area < 5000:
        return "Truck"
    else:
        return "Bus"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # **Step 1: Remove Background Noise**
    fgmask = fgbg.apply(gray)
    denoised = cv2.bitwise_and(gray, gray, mask=fgmask)

    # **Step 2: Edge Detection (After Removing Background)**
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # **Step 3: Contour Detection**
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **Step 4: Classify based on contours and display results**
    for contour in contours:
        # Ignore small contours that are noise
        if cv2.contourArea(contour) < 500:
            continue
        
        # Get bounding box around the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Classify the vehicle based on contour size
        vehicle_type = classify_vehicle(contour)

        # Draw bounding box and vehicle type label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # **Step 5: Display the frame with bounding boxes and classifications**
    cv2.imshow("Vehicle Detection and Classification", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
