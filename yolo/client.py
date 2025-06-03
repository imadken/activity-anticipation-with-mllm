import cv2
import requests
import numpy as np
from random import randint
from time import time

colors = [(128, 0, 128),(255, 165, 0),(0, 0, 255),(255, 0, 0)]
# API endpoint
API_URL = "http://10.12.20.79:8000/predict"  # Replace with your server's IP address
API_URL = "http://127.0.0.1:8000/predict"  # Replace with your server's IP address

# Initialize video capture (replace with your video path or 0 for webcam)

vid_path = "data/egocentric_vids_kousseila/scenevideo_kitchen.mp4"  # Replace with your video path
vid_path = "data/egocentric_vids_kousseila/scenevideo.mp4"  # Replace with your video path

cap = cv2.VideoCapture(vid_path)
# cap.set(cv2.CAP_PROP_FPS, 2)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")


count = 0

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    start = time()
    # Encode frame as JPEG
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()
    if count % 2 == 0:
    # Send frame to API
        try:
            response = requests.post(API_URL, files={"file": ("frame.jpg", img_bytes, "image/jpeg")})
            response.raise_for_status()  # Raise exception for bad status codes
    
            # Get YOLO results
            results = response.json()
            boxes = results["boxes"]
            scores = results["scores"]
            classes = results["classes"]
    
    
            # print(results["results"])
    
            # Process results (e.g., draw bounding boxes on frame)
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
    
                color = colors[randint(0,3)]
    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls}: {score:.2f}", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"Frame {count} processed in {time() - start:.2f} seconds")
            # Display the frame with results
            resized_frame = cv2.resize(frame, (1240, 720))
    
            #  Display the resized frame
            cv2.imshow("YOLO Results", resized_frame)
            
             
    
        except requests.RequestException as e:
            print(f"Error sending request to API: {e}")
            break
    count += 1
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()