import os
import cv2
import logging
import time # Penting untuk menghitung delay
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

model = YOLO('yolov8s.pt')

SAVE_DIR = "received_videos"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/upload/")
async def detect_objects(file: UploadFile = File(...)):
    # 1. Mulai hitung waktu upload/penerimaan file
    start_receive = time.time()
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov']:
        raise HTTPException(status_code=400, detail="Format video tidak didukung")

    temp_path = os.path.join(SAVE_DIR, f"process_{file.filename}")
    
    try:
        # Proses membaca file dan menulis ke disk
        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        # Hitung durasi upload
        upload_duration = time.time() - start_receive

        # 2. Mulai hitung waktu proses AI (Inference)
        start_ai = time.time()
        
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 
        
        max_frames = int(fps * 2) 
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
                break
            
            frame_count += 1

        cap.release()
        
        # Hitung durasi AI
        ai_duration = time.time() - start_ai
        
        detected_list = list(found_objects)
        logger.info(f"[SUCCESS] {file.filename} | Upload: {upload_duration:.3f}s | AI: {ai_duration:.3f}s")
        
        # 3. Kembalikan semua data waktu ke Client
        return {
            "status": "success",
            "filename": file.filename,
            "upload_time": upload_duration,   # Waktu terima file
            "inference_time": ai_duration,    # Waktu proses YOLO
            "detected": detected_list
        }

    except Exception as e:
        logger.error(f"[ERROR] Error: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)