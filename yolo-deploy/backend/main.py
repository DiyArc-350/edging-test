import os
import cv2
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

model = YOLO('yolov8n.pt')

SAVE_DIR = "received_videos"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/upload/")
async def detect_objects(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov']:
        raise HTTPException(status_code=400, detail="Format video tidak didukung")

    temp_path = os.path.join(SAVE_DIR, f"process_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 
        
        max_frames = int(fps * 2)  # Limit to 2 seconds
        frame_count = 0
        found_objects = set()

        while cap.isOpened() and frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)
            
            for r in results:
                for c in r.boxes.cls:
                    found_objects.add(model.names[int(c)])
            
            if found_objects:
                logger.info(f"[INFO] Detection found at frame {frame_count}")
                break
            
            frame_count += 1

        cap.release()
        
        detected_list = list(found_objects)
        logger.info(f"[SUCCESS] Selesai: {file.filename} | Terdeteksi: {detected_list}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "detected": detected_list,
            "processed_frames": frame_count
        }

    except Exception as e:
        logger.error(f"[ERROR] Error: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
