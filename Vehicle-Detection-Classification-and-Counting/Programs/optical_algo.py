import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Load video
video_path = r"Vehicle-Detection-Classification-and-Counting\Videos\video.mp4"
cap = cv2.VideoCapture(video_path)

# Background Subtraction for removing static objects
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Shi-Tomasi Corner Detection parameters
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=5, blockSize=7)

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
fgmask = fgbg.apply(old_gray)
old_gray = cv2.bitwise_and(old_gray, old_gray, mask=fgmask)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame_gray)
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=fgmask)

    if frame_count % 10 == 0 or p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and len(p1) > 0:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Compute displacement for each point
        displacements = np.linalg.norm(good_new - good_old, axis=1)
        motion_threshold = 2
        moving_new = good_new[displacements > motion_threshold]

        if len(moving_new) > 0:
            # **Cluster moving points using DBSCAN**
            clustering = DBSCAN(eps=30, min_samples=3).fit(moving_new)  # `eps` defines max distance between points
            labels = clustering.labels_

            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Ignore noise points
                    continue

                # Get all points in this cluster
                cluster_points = moving_new[labels == label]

                # Bounding box for cluster
                x_min, y_min = np.min(cluster_points, axis=0).astype(int)
                x_max, y_max = np.max(cluster_points, axis=0).astype(int)
                w, h = x_max - x_min, y_max - y_min

                # **Classify vehicle based on bounding box size**
                if w < 80 and h < 80:
                    vehicle_type = "Car"
                    color_box = (0, 255, 0)  # Green
                elif 80 <= w < 150 and h < 150:
                    vehicle_type = "Truck"
                    color_box = (0, 0, 255)  # Red
                else:
                    vehicle_type = "Bus"
                    color_box = (255, 0, 0)  # Blue

                # Draw single bounding box per cluster
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_box, 2)
                cv2.putText(frame, vehicle_type, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

        old_gray = frame_gray.copy()
        p0 = moving_new.reshape(-1, 1, 2)

    frame_count += 1
    cv2.imshow("Optical Flow Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
