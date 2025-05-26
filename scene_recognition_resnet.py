import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Load the model once
model_path = "models/places365/resnet152_places365.pth"  # Adjust if path differs
state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

model = models.resnet152(pretrained=False)
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    model.load_state_dict(state_dict["state_dict"])
else:
    model.load_state_dict(state_dict)
model.eval()

# Preprocess transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Places365 labels once
label_path = "data/places365/categories_places365.txt"
with open(label_path, "r") as f:
    classes = [line.strip().split()[0].split("/")[-1] for line in f]

def classify_scene(frame):
    """
    Classify an indoor scene from an image frame and annotate it with the top prediction.
    
    Args:
        frame (numpy.ndarray): Input image frame in BGR format (from OpenCV).
    
    Returns:
        tuple: (top_class, top_prob, annotated_frame)
            - top_class (str): The top predicted class.
            - top_prob (float): The probability of the top class.
            - annotated_frame (numpy.ndarray): Frame with the top label in the top-left corner.
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)

    # Preprocess
    input_tensor = preprocess(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 1 prediction
    top_prob, top_idx = probs.topk(1)
    top_class = classes[top_idx[0]]
    top_prob = float(top_prob[0])

    # Annotate frame
    annotated_frame = frame.copy()
    label = f"{top_class}: {top_prob:.3f}"
    cv2.putText(annotated_frame, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return top_class, top_prob, annotated_frame

# Example usage
if __name__ == "__main__":
    # Load an example frame
    img_path = "data/epickitchen/P01_101_frame_0000004863.jpg"
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to load image from {img_path}")
    else:
        # Classify and get results
        top_class, top_prob, annotated_frame = classify_scene(frame)
        print(f"Top prediction: {top_class}: {top_prob:.3f}")

        # Display the annotated frame
        cv2.imshow("Scene Classification", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()