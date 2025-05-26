import cv2
import mediapipe as mp
import numpy as np


def process_and_draw_hands(frame, hands):
    """
    Detects hands in a frame and draws bounding boxes around them.
    
    Args:
        frame (numpy.ndarray): Input frame in BGR format (from OpenCV).
        hands: MediaPipe Hands object for detection.
    
    Returns:
        numpy.ndarray: Frame with bounding boxes drawn (BGR format).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    bboxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            padding = 20
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
            bboxes.append((x_min, y_min, x_max, y_max))
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame_bgr, bboxes

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks (optional)
hands = mp_hands.Hands(
    static_image_mode=False,  # False for video (tracking mode)
    max_num_hands=2,         # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Video path
video_path = "path/to/your/video.mp4"  # Replace with your video path
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video resolution: {original_width}x{original_height}, FPS: {fps}")


# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    
    start_time = cv2.getTickCount()
    processed_frame,bbox = process_and_draw_hands(frame, hands)
    end_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (end_time - start_time)
    cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print(bbox)
    # Process frame with hand detection and draw bounding boxes
    # processed_frame = process_and_draw_hands(frame, hands)

    # Display the result
    cv2.imshow('Hand Detection with Bounding Boxes', processed_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()