import os
import cv2
import logging
import time
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model to GPU if available for lowest AI latency
model = YOLO('yolov8s.pt')
try:
    model.to('cuda')
except:
    logger.info("CUDA not available, using CPU.")

# Use RAM-disk for high-speed temporary storage
RAM_DISK = "/dev/shm/urllc_temp"
os.makedirs(RAM_DISK, exist_ok=True)

@app.post("/upload/")
async def detect_objects(file: UploadFile = File(...)):
    # 1. MEASURE UPLOAD/RECEPTION DELAY
    # We start timing as soon as the request hits the endpoint
    start_receive = time.perf_counter()
    
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(RAM_DISK, f"{uuid.uuid4()}{ext}")
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # End of reception/write phase
        upload_duration = time.perf_counter() - start_receive

        # 2. MEASURE AI INFERENCE TIME
        start_ai = time.perf_counter()
        
        cap = cv2.VideoCapture(temp_path)
        found_objects = set()
        
        # Process first 30 frames only to simulate real-time URLLC constraints
        for _ in range(30):
            success, frame = cap.read()
            if not success: break
            
            # imgsz=320 is optimized for speed
            results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)
            for r in results:
                for c in r.boxes.cls:
                    found_objects.add(model.names[int(c)])
            
            if found_objects: break # Exit early if object found to save time

        cap.release()
        ai_duration = time.perf_counter() - start_ai
        
        # 3. RETURN TIMING DATA TO CLIENT
        return {
            "status": "success",
            "upload_time": upload_duration,   # Time spent receiving/saving
            "inference_time": ai_duration,    # Time spent by YOLO
            "detected": list(found_objects)
        }

    except Exception as e:
        logger.error(f"Server Error: {e}")
        return {"status": "error", "upload_time": 0, "inference_time": 0, "detected": []}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)