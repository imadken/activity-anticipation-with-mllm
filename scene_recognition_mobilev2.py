import torch
from torchvision import models, transforms
import cv2
import numpy as np
import wandb
from PIL import Image

# Step 1: Initialize W&B and Download the Artifact
wandb.login()
run = wandb.init(project="places365-finetuning", job_type="inference")

# Download the fine-tuned model artifact
artifact = run.use_artifact('imadken/places365-finetuning/mobilenetv2-places365:v0', type='model')
artifact_dir = artifact.download()
model_path = f"{artifact_dir}/mobilenetv2_places365.pth"

# Step 2: Set Up Multi-GPU
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Initialize MobileNetV2
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 365)  # Adjust for Places365

# Load the fine-tuned weights (map to CPU initially to avoid issues)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Wrap with DataParallel for multi-GPU if available
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

# Move the model to GPU if available, otherwise keep on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Step 3: Preprocess Transformation
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 4: Load Places365 Labels
with open("data/places365/categories_places365.txt", "r") as f:
    classes = [line.strip().split()[0].split("/")[-1] for line in f]

# Step 5: Classify Scene Function (for both webcam and single image)
def classify_scene(frame):
    """
    Classify a scene from an image frame using fine-tuned MobileNetV2.
    
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
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return top_class, top_prob, annotated_frame

# Step 6: Function to Test with a Single Image
def test_single_image(image_path):
    """
    Test the model with a single image.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        None (prints results and displays the annotated image)
    """
    # Load and preprocess the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Classify the image
    top_class, top_prob, annotated_frame = classify_scene(frame)
    print(f"Top: {top_class}: {top_prob:.3f}")

    # Display the result
    cv2.imshow("Single Image Classification", annotated_frame)
    cv2.waitKey(0)  # Wait for any key press to close the window
    cv2.destroyAllWindows()

# Step 7: Main Execution with Option for Webcam or Single Image
if __name__ == "__main__":
    # Ask user for the mode
    print("Choose a mode:")
    print("1. Real-time webcam classification")
    print("2. Test with a single image")
    mode = input("Enter mode (1 or 2): ")

    if mode == "1":
        # Real-time webcam classification
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

    elif mode == "2":
        # Test with a single image
        # image_path = input("Enter the path to the image (e.g., 'path/to/image.jpg'): ")
        image_path =  ("data/epickitchen/P01_101_frame_0000004934.jpg")
        test_single_image(image_path)

    else:
        print("Invalid mode selected. Please choose 1 or 2.")

    # Finish the W&B run
    wandb.finish()