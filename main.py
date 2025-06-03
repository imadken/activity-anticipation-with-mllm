import os
import cv2
import numpy as np

from PIL import Image
from time import time

from utils3 import match_hands_to_objects, process_frame
# from scene_recognition_resnet import classify_scene



# Video path
video_path = "data/cooking-porridge-on-the-stove-a-mans-hand-stirs-the-porridge-with-a-spoon-in-a-SBV-347504834-preview.mp4"  # Replace with your video path
video_path = "data/egocentric_vids_kousseila/robot.mp4"  # Replace with your video path
video_path = "data/egocentric_vids_kousseila/Egocentric_desktop.mp4"  # Replace with your video path
video_path = "data/egocentric_vids_kousseila/scenevideo_kitchen.mp4"  # Replace with your video path
video_path = "data/egocentric_vids_kousseila/scenevideo.mp4"  # Replace with your video path

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# # Get video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.set(cv2.CAP_PROP_FPS, 2)  # Set FPS to 2 if needed
# fps = cap.get(cv2.CAP_PROP_FPS)



count = 0



while cap.isOpened():
    ret, frame = cap.read()
    

    if not ret:
        print("End of video or error reading frame.")
        break

    
    if count % 2:
        try:
            # Process frame with YOLO
            
            processed_frame, (yolo_bboxes, hand_bboxes) = process_frame(frame)
            # start = time()
            processed_frame = match_hands_to_objects(processed_frame, hand_bboxes, yolo_bboxes)
            # print(f"Frame {count} processed in {time() - start:.2f} seconds")
            

            # Verify processed_frame before display
            if not isinstance(processed_frame, np.ndarray):
                print("Error: Processed frame is not a NumPy array.")
                break
            if processed_frame.dtype != np.uint8:
                print(f"Error: Processed frame has incorrect dtype: {processed_frame.dtype}")
                break
            if processed_frame.shape[-1] != 3:
                print(f"Error: Processed frame has incorrect shape: {processed_frame.shape}")
                break

            # Display the frame
            resized_frame = cv2.resize(processed_frame, (1240, 720))

        
            cv2.imshow('ECAI2025', cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
            # last_frame_time = current_time

        except Exception as e:
            print(f"Error processing frame: {e}")
            break
     
    count += 1 
    
     # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
         print("Exiting... should implement saving db")
         break
    # print(count)

# Cleanup
# cap.release()
cv2.destroyAllWindows()
