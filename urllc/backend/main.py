import os
import cv2
import logging
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ENGINE CONFIGURATION: NanoDet-Plus ONNX Runtime
# ============================================================
MODEL_PATH = "nanodet-plus-m_320.onnx"

# Configure ONNX Runtime to prioritize sequential CPU processing efficiency
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# OPTIMIZATION: Completely mutes graph optimization warnings in terminal logs
opts.log_severity_level = 3

if not os.path.exists(MODEL_PATH):
    logger.critical(f"Model file '{MODEL_PATH}' missing! Please check your Docker build layers.")

ort_session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

# NanoDet COCO standard 80 classes map
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def preprocess_binary_frame(frame_bytes, input_shape=(320, 320)):
    """Decodes raw binary network packets straight into a normalized matrix buffer"""
    # 1. Convert raw socket bytes into a NumPy uint8 array
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    
    # 2. Decode the compressed JPEG/PNG bytes straight into pixel matrices in RAM
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None
        
    orig_h, orig_w = frame.shape[:2]
    
    # 3. Resize and normalize for NanoDet-Plus
    img = cv2.resize(frame, input_shape)
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    img = (img.astype(np.float32) - mean) / std
    
    # 4. HWC image layout matrix transpose to CHW
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0), orig_w, orig_h

# ============================================================
# WEBSOCKET ROUTE
# ============================================================
@app.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Handles high-frequency persistent bi-directional streaming threads"""
    await websocket.accept()
    logger.info("Client connected via secure persistent streaming pipe.")
    
    try:
        while True:
            # 1. Pull raw binary frame slice directly out of network interface card
            frame_bytes = await websocket.receive_bytes()
            start_pipeline = time.perf_counter()
            
            # 2. Process image matrix entirely inside volatile system RAM
            blob, w, h = preprocess_binary_frame(frame_bytes)
            if blob is None:
                await websocket.send_json({"status": "error", "message": "Corrupted frame array format"})
                continue
                
            # 3. Direct low-latency C++ matrix graph evaluation pass via ONNX Runtime
            outputs = ort_session.run(None, {input_name: blob})
            raw_output = outputs[0][0]  # Array dimensions mapping to [Candidates, Attributes]
            
            # 4. FIXED: Explicitly isolate the 80 categorical classification score columns 
            # to protect boundaries from the bounding box regression attributes.
            scores_matrix = raw_output[:, :80]
            class_ids = np.argmax(scores_matrix, axis=-1)
            max_scores = np.max(scores_matrix, axis=-1)
            
            detected = set()
            for idx, score in enumerate(max_scores):
                if score > 0.45:  # Classification confidence limit threshold
                    class_idx = int(class_ids[idx])
                    if class_idx < len(CLASSES):  # Defensive boundaries guard
                        detected.add(CLASSES[class_idx])
            
            latency_ms = (time.perf_counter() - start_pipeline) * 1000
            
            # 5. Instantly shoot a JSON packet back to the client while connection stays alive
            await websocket.send_json({
                "status": "success",
                "inference_time_ms": round(latency_ms, 2),
                "detected": list(detected)
            })
            
    except WebSocketDisconnect:
        logger.info("Client cleanly closed the real-time pipeline connection socket.")
    except Exception as e:
        logger.error(f"WebSocket execution exception caught: {e}")

# ============================================================
# APPLICATION ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # Enforces string-import mapping format to guarantee multi-worker stability
    uvicorn.run("main:app", host="0.0.0.0", port=80, workers=2)
