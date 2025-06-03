from fastapi import FastAPI, HTTPException, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn


app = FastAPI(title="YOLOv8 Frame Inference API")

#
model = YOLO("models/yolov8-custom_dataset/weights/best_yolo8_custom.pt", verbose=False)

@app.get("/hello")
async def startup_event():
    """Event handler for application startup."""
    return {"message": "YOLOv8 Frame Inference API is running"}
    

@app.post("/predict")
async def predict(file: bytes = File(...)):
    try:
        # Decode CV2 frame from bytes
        nparr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")

        # Run YOLOv8 inference
        # results = model.predict(source=frame, conf=0.25, iou=0.45)
        results = model(frame)
        
    
        # # Extract results
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()  # Bounding box coordinates
        scores = results[0].boxes.conf.cpu().numpy().tolist()  # Confidence scores
        classes = results[0].boxes.cls.cpu().numpy().tolist()  # Class IDs
        class_names = [model.names[int(cls_id)] for cls_id in classes]

        return JSONResponse(content={
            "boxes": boxes,
            "scores": scores,
            "classes": class_names
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)