import torch
import torchvision.transforms as T
import cv2
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor, ViTForImageClassification, ViTImageProcessor

# Load DETR model for vehicle detection
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Load ViT model for vehicle classification
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Define COCO classes (DETR is trained on COCO dataset)
COCO_CLASSES = ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]

# Open video file
video_path = "Vehicle-Detection-Classification-and-Counting/Videos/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor for DETR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = detr_processor(images=image, return_tensors="pt")

    # Perform detection
    with torch.no_grad():
        outputs = detr_model(**inputs)

    # Extract detections
    probas = outputs.logits.softmax(-1)[0, :, :-1]  # Exclude background class
    keep = probas.max(-1).values > 0.7  # Keep only high-confidence detections

    boxes = outputs.pred_boxes[0, keep].detach().cpu().numpy()
    labels = probas.argmax(-1)[keep].detach().cpu().numpy()

    for (box, label) in zip(boxes, labels):
        if COCO_CLASSES[label] in ["car", "bus", "truck"]:
            x_center, y_center, width, height = box
            x1, y1 = int((x_center - width / 2) * frame.shape[1]), int((y_center - height / 2) * frame.shape[0])
            x2, y2 = int((x_center + width / 2) * frame.shape[1]), int((y_center + height / 2) * frame.shape[0])

            # Crop detected vehicle for classification
            vehicle_crop = frame[y1:y2, x1:x2]
            vehicle_crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)

            # Process crop for ViT classification
            inputs = vit_processor(images=vehicle_crop_rgb, return_tensors="pt")

            # Predict class using ViT
            with torch.no_grad():
                vit_outputs = vit_model(**inputs)
            predicted_class = vit_outputs.logits.argmax(-1).item()

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{COCO_CLASSES[label]} - Class {predicted_class}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("DETR + ViT Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
