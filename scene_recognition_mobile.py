import torch
from torchvision import models, transforms
import cv2
import numpy as np

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MobileNetV3-Small (ImageNet pretrained)
model = models.mobilenet_v3_small(pretrained=True)
model.eval().to(device)

# Preprocess transformation
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet labels (since pretrained is ImageNet)
# Download from: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
with open("data\imagenet\imagenet1000_clsidx_to_labels.txt", "r") as f:
    classes = [line.strip() for line in f]

def classify_scene(frame):
    """
    Classify a scene from an image frame using MobileNetV3-Small.
    
    Args:
        frame (numpy.ndarray): Input BGR frame from OpenCV.
    
    Returns:
        tuple: (top_class, top_prob, annotated_frame)
    """
    # Fast resize with OpenCV
    frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output = model(input_tensor)
        else:
            output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    # Top 1 prediction
    top_prob, top_idx = probs.topk(1)
    top_class = classes[top_idx[0]]
    top_prob = float(top_prob[0])

    # Annotate frame
    annotated_frame = frame.copy()
    label = f"{top_class}: {top_prob:.3f}"
    cv2.putText(annotated_frame, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)

    return top_class, top_prob, annotated_frame

# Real-time example
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Webcam or video file
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Subsample: classify every 5th frame
        if frame_count % 5 == 0:
            top_class, top_prob, annotated_frame = classify_scene(frame)
            print(f"Top: {top_class}: {top_prob:.3f}")
        else:
            annotated_frame = frame
        
        cv2.imshow("Scene Classification", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()