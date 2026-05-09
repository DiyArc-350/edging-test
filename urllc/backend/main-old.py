import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import logging

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
            shutil.copyfileobj(file.file, buffer)

        # imgsz=320 untuk kecepatan, stream=True untuk hemat RAM
        results = model.predict(source=temp_path, conf=0.5, imgsz=320)
        
        found_objects = set()
        for r in results:
            for c in r.boxes.cls:
                found_objects.add(model.names[int(c)])
        
        detected_list = list(found_objects)
        logger.info(f"[SUCCESS] Selesai: {file.filename} | Terdeteksi: {detected_list}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "detected": detected_list
        }

    except Exception as e:
        logger.error(f"[ERROR] Error: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
