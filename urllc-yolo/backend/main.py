import os
import cv2
import logging
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ENGINE CONFIGURATION: YOLO11 Nano ONNX Setup
# ============================================================
# Using the optimized, lightning-fast ONNX matrix model compilation
MODEL_PATH = "yolo11n.onnx"

if not os.path.exists(MODEL_PATH):
    logger.info(f"Target asset '{MODEL_PATH}' not found locally. Initializing compile sequence...")
    # Safe fallback: Pull weights and compile to ONNX matching our 320x320 pipeline resolution
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    pt_model = YOLO("yolo11n.pt")
    pt_model.export(format="onnx", imgsz=320, half=True)

# Load the compiled ONNX graph engine directly into memory
model = YOLO(MODEL_PATH, task="detect")

# ============================================================
# HELPER FUNCTIONS (Pure In-Memory Handling)
# ============================================================
def decode_binary_frame(frame_bytes):
    """
    Decodes incoming raw socket packet bytes straight into an image matrix buffer.
    Bypasses disk storage completely to protect hardware from write wear.
    """
    # Convert raw bytes block directly to a NumPy array buffer
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    
    # Decode compressed JPEG/PNG bytes straight to raw BGR pixels in RAM
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

# ============================================================
# WEBSOCKET STREAMING ENDPOINT
# ============================================================
@app.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Handles persistent, real-time bi-directional streaming connection sockets"""
    await websocket.accept()
    logger.info("Client connected via high-speed YOLO persistent stream pipe.")
    
    try:
        while True:
            # 1. Pull raw compressed JPEG frame slice straight out of network interface card
            frame_bytes = await websocket.receive_bytes()
            start_pipeline = time.perf_counter()
            
            # 2. Decode bytes matrix entirely within volatile system RAM
            frame = decode_binary_frame(frame_bytes)
            if frame is None:
                await websocket.send_json({"status": "error", "message": "Corrupted frame matrix buffer"})
                continue
                
            # 3. Direct low-latency frame evaluation pass through the ONNX graph
            results = model.predict(
                source=frame,
                conf=0.45,       # Confidence threshold optimization limit
                imgsz=320,       # Match image resolution constraints
                verbose=False,   # Mutes internal console logging overhead
                half=True        # Employs optimized FP16 half-precision calculations if supported
            )
            
            # 4. Extract unique classifications from predictions vector
            detected = set()
            for r in results:
                for c in r.boxes.cls:
                    detected.add(model.names[int(c)])
            
            latency_ms = (time.perf_counter() - start_pipeline) * 1000
            
            # 5. Instantly shoot JSON payload back to client while the TCP link stays open
            await websocket.send_json({
                "status": "success",
                "inference_time_ms": round(latency_ms, 2),
                "detected": list(detected)
            })
            
    except WebSocketDisconnect:
        logger.info("Client cleanly terminated the real-time YOLO streaming pipeline connection socket.")
    except Exception as e:
        logger.error(f"WebSocket execution exception caught: {e}")

# ============================================================
# RUN CONTEXT ENTRY
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # Enforces explicit string-import mapping to avoid multi-worker subprocess duplication errors
    uvicorn.run("main:app", host="0.0.0.0", port=80, workers=2)