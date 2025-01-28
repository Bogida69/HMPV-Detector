# import the necessary packages
from ultralytics import YOLO
import numpy as np
import argparse
import imutils
import cv2
import os
from scipy.spatial import distance as dist

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# Load YOLOv5 model (small model by default, use yolov5m/yolov5l/yolov5x for bigger models)
print("[INFO] loading YOLOv5 model...")
model = YOLO('yolo11m.pt')  # You can replace it with yolov5m.pt, yolov5l.pt, yolov5x.pt

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames from the video stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # resize the frame for better performance
    frame = imutils.resize(frame, width=700)

    # Run YOLOv5 inference
    results = model(frame)

    # Extract bounding boxes and centroids
    boxes = results[0].boxes  # Accessing the first result (there could be multiple, if batching)

    centroids = []
    person_count = 0  # Counter to track the number of people detected

    # Initialize the set of indexes that violate social distancing
    violate = set()

    # loop over each bounding box and extract coordinates and centroids
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Accessing the bounding box in xyxy format
        cX = (x1 + x2) // 2
        cY = (y1 + y2) // 2
        centroids.append((cX, cY))

        # Check if the label corresponds to 'person' class (class 0 is 'person' in COCO dataset)
        if box.cls == 0:  # 'person' class in YOLO model
            person_count += 1

    # Compute Euclidean distances between centroids and check for violations
    if len(centroids) >= 2:
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < 100:  # Example: minimum distance in pixels (you can tune this)
                    violate.add(i)
                    violate.add(j)

    # Loop over the results and draw bounding boxes and violations
    for i, (box, centroid) in enumerate(zip(boxes, centroids)):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Same here for accessing coordinates
        (cX, cY) = centroid

        # Convert centroid to integers
        cX, cY = int(cX), int(cY)

        # Set color based on violation
        color = (0, 255, 0)  # green for no violation
        if i in violate:
            color = (0, 0, 255)  # red for violation

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw centroid circle
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # Display the result with social distancing violations and the count of people
    text = f"Social Distancing Violations: {len(violate)} | People Count: {person_count}"
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # Show the output frame if display is enabled
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Write to output video if a file path is provided
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)
