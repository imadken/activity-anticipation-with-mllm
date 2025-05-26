import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

from utils2 import match_hands_to_objects, process_frame, hands
# from scene_recognition_resnet import classify_scene





# Initialize YOLOv8
yolo_model = YOLO("yolov8l.pt",verbose=False)  # Nano model for speed; use yolov8s/m/l for better accuracy
#initialize the database



#if reading frames from a directory
frames_dir = "data/epickitchen"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg'))])

# Video path
video_path = "data/cooking-porridge-on-the-stove-a-mans-hand-stirs-the-porridge-with-a-spoon-in-a-SBV-347504834-preview.mp4"  # Replace with your video path
video_path = "data/egocentric_vids_kousseila/robot.mp4"  # Replace with your video path
video_path = "data/egocentric_vids_kousseila/Egocentric.mp4"  # Replace with your video path

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# # Get video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# print(f"Video resolution: {original_width}x{original_height}, FPS: {fps}")

# Main loop to process video frames
# for frame_file in frame_files:
#     # Read the frame
#      frame_path = os.path.join(frames_dir, frame_file)
#      frame = cv2.imread(frame_path)
#      sleep(0.2)
# with tqdm(desc="processing frames", unit="frames") as pbar:
count = 0

while cap.isOpened():
    
     ret, frame = cap.read()
     if not ret:
         print("End of video or error reading frame.")
         break
 
     # Process frame with YOLO and MediaPipe
     try:
         
         processed_frame, (yolo_bboxes, hand_bboxes) = process_frame(frame, yolo_model, hands)

         
        

         processed_frame = match_hands_to_objects(processed_frame, hand_bboxes, yolo_bboxes)
        #  cls,pred , processed_frame= classify_scene(processed_frame)
         
         count += 1
        #  if (count % 5) == 0:
            # cls,pred , processed_frame= classify_scene(processed_frame)
            #  cls = get_response(["what is the location ? Return only the answer",Image.fromarray(processed_frame)])
            #  cls = get_response(["what is the objects in hands ? Return only the answer as well as the bounding box in the following format: (object, [coordinates])",Image.fromarray(processed_frame)])
             
            # print(f"Scene is: {cls}  , count {count}")
             
            # Classify scene
        #  cls,pred , processed_frame= classify_scene(processed_frame)

         
        #  print("YOLO bboxes:", yolo_bboxes)
        #  print("Hand bboxes:", hand_bboxes)
     except Exception as e:
         print(f"Error processing frame: {e}")
         break
 
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
 
     # Display 
    #  cv2.imshow('YOLO + Hand Detection', processed_frame)
     cv2.imshow('YOLO + Hand Detection', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
     
     
    #  pbar.update(1)
     # Press 'q' to quit
     if cv2.waitKey(1) & 0xFF == ord('q'):
         print("Exiting... should implement saving db")
         break
     print(count)
# Cleanup
# cap.release()
cv2.destroyAllWindows()
hands.close()