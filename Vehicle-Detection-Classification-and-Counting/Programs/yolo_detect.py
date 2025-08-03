import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' is the smallest model; use 'm' or 'l' for better accuracy

# Open video file
video_path = "Vehicle-Detection-Classification-and-Counting/Videos/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cnt_up = 0
cnt_down = 0
line_up = 400
line_down = 250
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (900, 500))
    
    # Perform YOLOv8 detection
    results = model(frame)
    detections = results[0].boxes.data  # Bounding box data
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        class_id = int(cls)
        label = model.names[class_id]  # Get class label
        
        # Draw bounding box
        color = (0, 255, 0) if "car" in label.lower() else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), font, 0.5, color, 2, cv2.LINE_AA)
        
        cy = (y1 + y2) / 2  # Get center y-coordinate
        if cy > line_down and cy < line_up:
            if cy < (line_up + line_down) / 2:
                cnt_up += 1
            else:
                cnt_down += 1
    
    # Draw counting lines
    cv2.line(frame, (0, line_up), (900, line_up), (255, 0, 255), 3)
    cv2.line(frame, (0, line_down), (900, line_down), (255, 0, 0), 3)
    
    # Display counts
    cv2.putText(frame, f'UP: {cnt_up}', (10, 40), font, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f'DOWN: {cnt_down}', (10, 90), font, 0.5, (255, 0, 0), 2)
    
    cv2.imshow("YOLOv8 Vehicle Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
