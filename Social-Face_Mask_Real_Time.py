import argparse
import imutils
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# Load the YOLOv5 model for crowd detection (small model for faster processing)
print("[INFO] loading YOLO model for crowd detection...")
crowd_model = YOLO('yolo11m.pt')  # Load the crowd detection model

# Load the YOLO model for mask detection (you should provide your own trained model)
print("[INFO] loading YOLO model for mask detection...")
mask_model = YOLO(r'C:\Drive E\me\Python_for_ML\Personal_Projects\HMPV\Face_Mask_Dataset\Result\runs\detect\train\weights\best.pt')  # Mask detection model

# Initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# Set the confidence threshold for detection (only persons with a higher confidence score will be considered)
confidence_threshold = 0.6  # Adjust confidence to filter out non-person objects like phones

# Loop over the frames from the video stream
while True:
    # Read the next frame from the file
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # Resize the frame for better performance
    frame = imutils.resize(frame, width=700)

    # Run YOLOv5 inference for crowd detection
    crowd_results = crowd_model(frame)

    # Run YOLOv5 inference for mask detection
    mask_results = mask_model(frame)[0]

    # Extract crowd bounding boxes and centroids
    crowd_boxes = crowd_results[0].boxes
    centroids = []
    person_count = 0  # Counter to track the number of people detected
    violate = set()  # Set to track violations for social distancing

    # Loop through each detected object in the crowd model output
    for box in crowd_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Accessing the bounding box
        cX = (x1 + x2) // 2
        cY = (y1 + y2) // 2
        centroids.append((cX, cY))

        # Check if the detected class is person (class_id = 0 corresponds to 'person' in YOLOv5)
        if box.cls == 0 and box.conf >= confidence_threshold:  # Only consider persons with confidence above threshold
            person_count += 1

    # Compute Euclidean distances between centroids and check for violations
    if len(centroids) >= 2:
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < 100:  # Minimum distance in pixels for social distancing
                    violate.add(i)
                    violate.add(j)

    # Initialize a flag to check if any violations are detected
    show_violation_boxes = False

    # Iterate over the results of the mask detection
    for result in mask_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > confidence_threshold:  # Confidence threshold for mask detection
            # Always draw the bounding box for mask detection (green color)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Get class label (assuming 0: 'mask', 1: 'no mask')
            label = 'Mask' if class_id == 0 else 'No Mask'

            # Annotate the class label
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Draw the bounding boxes for people detected and highlight violations
    for i, (box, centroid) in enumerate(zip(crowd_boxes, centroids)):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        (cX, cY) = centroid

        # Convert centroid to integers
        cX, cY = int(cX), int(cY)

        # Set color based on violation (Red for violation)
        if i in violate:
            show_violation_boxes = True  # Set the flag to True when a violation occurs
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red box for violations

    # Display the result with social distancing violations and the count of people
    text = f"Social Distancing Violations: {len(violate)} | People Count: {person_count}"
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # Show only the mask bounding boxes, and add violation boxes only if needed
    if show_violation_boxes:
        cv2.imshow("Frame", frame)

    # Show the output frame if display is enabled
    if args["display"] > 0 and not show_violation_boxes:
        # Show the frame with only mask detection bounding boxes
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

# Release resources
vs.release()
cv2.destroyAllWindows()
