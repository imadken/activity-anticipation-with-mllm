import cv2
from Rag import VideoFrameVectorDB
from PIL import Image
import numpy as np
from random import randint


labels = ["person", "car", "cat", "dog"]  # Example labels for demonstration

print("Initializing VideoFrameVectorDB...")

# Initialize the database (in-memory for demo)
db = VideoFrameVectorDB(collection_name="test_collection1", persistent_path="persistent_paths/test_persistent_path")
                 

# Example: Process a live video feed (e.g., webcam)
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file/stream URL
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)

    # Insert frame with a label (example: label every 10th frame)
    if frame_count % 10 == 0:
        label = labels[randint(0, len(labels) - 1)]
        frame_id = db.insert_frame(frame=pil_frame, label=label)
        print(f"Inserted frame with ID: {frame_id}")

    # Example: Search for similar frames every 30 frames
    if frame_count % 30 == 0:
        results = db.search_similar_frames(query_frame=pil_frame, n_results=1, label_filter="person")
        for result in results:
            print(f"ID: {result['id']}, Label: {result['label']}, Distance: {result['distance']}")
    # Display the frame
    cv2.imshow('Video Feed', frame)


    frame_count += 1
    # Press 'q' to quit (for webcam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()