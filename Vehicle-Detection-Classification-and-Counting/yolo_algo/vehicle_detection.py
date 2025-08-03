import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' is the smallest model; change to 'm' or 'l' for better accuracy

# Open video file
cap = cv2.VideoCapture(r"C:\Users\prem\Documents\22CSE560\COMPUTER VISION\Vehicle-Detection-Classification-and-Counting\Programs\Videos\video.mp4")

# Check if the video file is opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 detection
    results = model(frame)

    # Draw detection results
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
