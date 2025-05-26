from scipy.spatial import KDTree
import cv2
import numpy as np
import mediapipe as mp
from gemini_api import get_response_google
from PIL import Image
from Rag import VideoFrameVectorDB
import uuid

db = VideoFrameVectorDB(collection_name="ECAI25", persistent_path="persistent_paths/Ecai25_persistent_path")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

exclude_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe','traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench','dining table']  # Classes to exclude from detection

def detect_objects_yolo(frame, yolo_model):
    """
    Detects objects in a frame using YOLOv8 and draws bounding boxes.
    
    Args:
        frame (numpy.ndarray): Input frame in BGR format (from OpenCV).
        yolo_model: YOLOv8 model for object detection.
    
    Returns:
        numpy.ndarray: Frame with YOLO bounding boxes drawn (BGR format).
        list: List of bounding boxes [(x1, y1, x2, y2), ...].
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("Input frame must be a NumPy array.")
    
    
    # Convert BGR to RGB (YOLOv8 expects RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    org_frame= frame.copy()
  
    # YOLO object detection
    yolo_results = yolo_model(frame)
    yolo_bboxes = {}

    # Draw YOLO bounding boxes
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            cls_id = int(box.cls[0].item())
            cls_name = yolo_model.names[cls_id]
    
            # Skip all excluded classes
            if cls_name in exclude_classes:
                continue 

            label = f"{cls_name} {conf:.2f}"
            # label = f"{yolo_model.names[cls]}"

            if float(conf) < 0.8:  
                
                similarity = db.search_similar_frames(query_frame=Image.fromarray(org_frame[y1:y2 , x1:x2]), n_results=1)

                if similarity != []:
                   print(f"Similarity score of {cls_name} is {similarity[0]['distance']} with a label of {similarity[0]['label']} ")
               
                #    if similarity[0]['distance'] > 0.4 or (cls_name.lower() == similarity[0]['label']):
                   if similarity[0]['distance'] > 0.4:
                       label = similarity[0]['label']   
                       db.insert_frame(frame=Image.fromarray(org_frame[y1:y2 , x1:x2]), label=label.lower().strip(), id=None)
                   else:
                       
                        print(f"similarity is low, MLLM call..")
                        label = get_response_google(["what is this object ? return only the answer in one word if possible.",Image.fromarray(org_frame[y1:y2 , x1:x2])]).replace("?", "")
                        print(f"The MLLM response is: {label.lower()}")

                        db.insert_frame(frame=Image.fromarray(org_frame[y1:y2 , x1:x2]), label=label.lower().strip(), id=None)
                           
                       
                else: ## similarity empty, vectordb is not filled yet
                    print(f"similarity is empty, MLLM call..")
                    label = get_response_google(["what is this object ? return only the answer in one word if possible.",Image.fromarray(frame)]).replace("?", "")
                    print(f"The MLLM response is: {label}")
   
                    db.insert_frame(frame=Image.fromarray(org_frame[y1:y2 , x1:x2]), label=label.lower().strip(), id=None)
 
                
            yolo_bboxes[label]=(x1, y1, x2, y2)
            
            # Draw YOLO bounding box (blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label.replace("?", "").strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            

    return frame, yolo_bboxes



def detect_hands_mediapipe(frame, hands):
    """
    Detects hands in a frame using MediaPipe and draws bounding boxes.
    
    Args:
        frame (numpy.ndarray): Input frame in BGR format (from OpenCV).
        hands: MediaPipe Hands object for detection.
    
    Returns:
        numpy.ndarray: Frame with hand bounding boxes drawn (BGR format).
        list: List of hand bounding boxes [(x_min, y_min, x_max, y_max), ...].
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("Input frame must be a NumPy array.")

    # Convert BGR to RGB (MediaPipe expects RGB)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # hand_results = hands.process(frame_rgb)
    hand_results = hands.process(frame)
    hand_bboxes = []

    # Draw MediaPipe hand bounding boxes
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            
            # Find min/max coordinates from landmarks
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            
            # Add padding to the bounding box
            padding = 20
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
            hand_bboxes.append((x_min, y_min, x_max, y_max))
            
            # Draw hand bounding box (green)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "hand", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Optional: Draw landmarks and connections
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame, hand_bboxes

def process_frame(frame, yolo_model, hands):
    """
    Combines YOLO object detection and MediaPipe hand detection on a frame.
    
    Args:
        frame (numpy.ndarray): Input frame in BGR format (from OpenCV).
        yolo_model: YOLOv8 model for object detection.
        hands: MediaPipe Hands object for hand detection.
    
    Returns:
        numpy.ndarray: Frame with all bounding boxes drawn (BGR format).
        tuple: (yolo_bboxes, hand_bboxes) containing lists of bounding boxes.
    """
    # Create a copy of the frame to avoid modifying the original
    frame_copy = frame.copy()

    # Run YOLO object detection
    frame_copy, yolo_bboxes = detect_objects_yolo(frame_copy, yolo_model)

    # Run MediaPipe hand detection
    frame_copy, hand_bboxes = detect_hands_mediapipe(frame_copy, hands)

    return frame_copy, (yolo_bboxes, hand_bboxes)


def match_hands_to_objects(frame, hands, objects):
    """
    Match each hand to its nearest object and draw a line between their centroids.
    
    Args:
        frame: Input image/frame (numpy array in BGR format)
        hands: List of hand bounding boxes [(x_min, y_min, x_max, y_max), ...]
        objects: Dictionary of object bounding boxes  {label: (x_min, y_min, x_max, y_max), ...}
    
    Returns:
        frame: Modified frame with lines drawn
    """
    if not hands :
        return frame 
    
    if not objects:

        #implement MLLM call
        print("No objects detected, MLLM call should be implemented to return label as well as bounding box")
        # response = get_response_google(["what is the objects in hands ? Return only the answer as well as the bounding box in the following format: (object, [coordinates])",Image.fromarray(frame)])
        # print(f"The response is: {response}")
        return frame  
    

    # Compute centroids for hands
    hand_centroids = []
    for (x_min, y_min, x_max, y_max) in hands:
        centroid_x = (x_min + x_max) // 2
        centroid_y = (y_min + y_max) // 2
        hand_centroids.append((centroid_x, centroid_y))

    # Compute centroids for objects
    object_centroids = []
    object_labels = list(objects.keys())  # Keep track of labels for reference
    for bbox in objects.values():
        x_min, y_min, x_max, y_max = bbox
        centroid_x = (x_min + x_max) // 2
        centroid_y = (y_min + y_max) // 2
        object_centroids.append((centroid_x, centroid_y))

    # Convert to numpy arrays for KDTree
    hand_centroids = np.array(hand_centroids)
    object_centroids = np.array(object_centroids)

    # Build KDTree with object centroids
    tree = KDTree(object_centroids)

    # Find nearest object for each hand
    distances, indices = tree.query(hand_centroids)

    # Draw lines and bounding boxes
    OOI=[]
    for i, hand_bbox in enumerate(hands):
        # Hand centroid
        hand_x_min, hand_y_min, hand_x_max, hand_y_max = hand_bbox
        hand_centroid = (hand_centroids[i][0], hand_centroids[i][1])

        # Nearest object centroid
        nearest_idx = indices[i]
        nearest_object_bbox = list(objects.values())[nearest_idx]
        obj_x_min, obj_y_min, obj_x_max, obj_y_max = nearest_object_bbox
        nearest_object_centroid = (object_centroids[nearest_idx][0], object_centroids[nearest_idx][1])

        
        # Draw line between hand and nearst object
        cv2.line(frame, hand_centroid, nearest_object_centroid, (0, 255, 0), 2)

        # Optional: Draw centroids
        cv2.circle(frame, hand_centroid, 5, (255, 0, 0), -1)  # Blue dot for hand
        cv2.circle(frame, nearest_object_centroid, 5, (0, 0, 255), -1)  # Red dot for object

        ooi = object_labels[nearest_idx]


        OOI.append(ooi.strip())

    Image.fromarray(frame).save(f"rag_images/frames/{str(uuid.uuid4())}.jpg")    

    print("OOI:", OOI)
    # for i, ooi in enumerate(OOI,start=1):
     
    #   print(f"OOI 0{i} is {ooi[:-5]} with conf: {ooi[-5:]}")
    
    return frame
