import os
import socket
import struct
import cv2
import numpy as np
import time
import logging
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Must match the client's structural header configurations exactly
MAX_PACKET_SIZE = 1500
HEADER_FORMAT = "!IIHH" # Frame Index (4B), Chunk Index (4B), Total Chunks (2B), Data Length (2B)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# ============================================================
# ENGINE CONFIGURATION: NanoDet-Plus ONNX Setup
# ============================================================
MODEL_PATH = "nanodet-plus-m_320.onnx"

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.log_severity_level = 3  # Mute warnings completely

if not os.path.exists(MODEL_PATH):
    logger.critical(f"Model file '{MODEL_PATH}' missing!")

ort_session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

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
# IN-MEMORY PREPROCESSING
# ============================================================
def preprocess_matrix(frame, input_shape=(320, 320)):
    orig_h, orig_w = frame.shape[:2]
    img = cv2.resize(frame, input_shape)
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    img = (img.astype(np.float32) - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0), orig_w, orig_h

# ============================================================
# MAIN UDP SOCKET STREAM RECEIVER LOOP
# ============================================================
def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(("0.0.0.0", 9998)) # Bound internally to Port 9998
    logger.info("URLLC NanoDet UDP Inference Server listening on port 9998...")

    buffer = {}  # Temporary storage to reconstruct matching frame IDs

    while True:
        try:
            packet, addr = server_socket.recvfrom(MAX_PACKET_SIZE)
            if len(packet) < HEADER_SIZE:
                continue

            header = packet[:HEADER_SIZE]
            frame_idx, chunk_idx, total_chunks, data_len = struct.unpack(HEADER_FORMAT, header)
            chunk_data = packet[HEADER_SIZE:HEADER_SIZE + data_len]

            if frame_idx not in buffer:
                buffer[frame_idx] = {"chunks": {}, "total": total_chunks, "timestamp": time.perf_counter()}

            buffer[frame_idx]["chunks"][chunk_idx] = chunk_data

            # If all fragments for this frame have arrived successfully
            if len(buffer[frame_idx]["chunks"]) == total_chunks:
                start_pipeline = time.perf_counter()
                
                # Chronologically merge fragments entirely in RAM
                full_frame_bytes = b"".join([buffer[frame_idx]["chunks"][i] for i in range(total_chunks)])
                
                np_arr = np.frombuffer(full_frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    blob, w, h = preprocess_matrix(frame)
                    
                    # Compute ONNX Runtime inference pass
                    outputs = ort_session.run(None, {input_name: blob})
                    raw_output = outputs[0][0]
                    
                    # Isolate scores tensor from bounding box indices (protect boundary parameters)
                    scores_matrix = raw_output[:, :80]
                    class_ids = np.argmax(scores_matrix, axis=-1)
                    max_scores = np.max(scores_matrix, axis=-1)
                    
                    detected = set()
                    for idx, score in enumerate(max_scores):
                        if score > 0.45:
                            class_idx = int(class_ids[idx])
                            if class_idx < len(CLASSES):
                                detected.add(CLASSES[class_idx])
                    
                    latency_ms = (time.perf_counter() - start_pipeline) * 1000
                    print(f"[NANODET FRAME {frame_idx:03d}] Processing: {latency_ms:.2f}ms | Objects: {list(detected)}")
                
                del buffer[frame_idx]

            # Periodic garbage collection for fragments broken by wireless drops
            current_time = time.perf_counter()
            old_frames = [fid for fid, f_data in buffer.items() if current_time - f_data["timestamp"] > 1.0]
            for fid in old_frames:
                del buffer[fid]

        except Exception as e:
            logger.error(f"Execution Error: {e}")

if __name__ == "__main__":
    main()