import os
import socket
import struct
import cv2
import numpy as np
import time
import json
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
# LIVE SERVER PERFORMANCE STATS WIDGET
# ============================================================
server_metrics = {
    "packets_received": 0,
    "frames_attempted": set(),
    "frames_reassembled": 0,
    "frames_dropped": 0,
    "total_inference_time": 0.0
}

def print_server_session_summary():
    """Prints a structured analytics report of the edge receiver's network health"""
    total_attempts = len(server_metrics["frames_attempted"])
    if total_attempts == 0:
        return
        
    success_rate = (server_metrics["frames_reassembled"] / total_attempts) * 100
    avg_ai = (server_metrics["total_inference_time"] / server_metrics["frames_reassembled"]) if server_metrics["frames_reassembled"] > 0 else 0
    
    print("\n" + "=" * 60)
    print("      EDGE SERVER REAL-TIME TRAFFIC & PROCESSING REPORT")
    print("-" * 60)
    print(f" Total Raw UDP Datagrams Processed : {server_metrics['packets_received']}")
    print(f" Total Unique Frames Attempted     : {total_attempts}")
    print(f" Successfully Reassembled Frames  : {server_metrics['frames_reassembled']}")
    print(f" Corrupted / Dropped Frames       : {server_metrics['frames_dropped']}")
    print(f" Frame Reassembly Success Rate     : {success_rate:.2f}%")
    print(f" Average Model Inference Latency   : {avg_ai:.2f} ms")
    print("=" * 60 + "\n")

# ============================================================
# IN-MEMORY PREPROCESSING
# ============================================================
def preprocess_matrix(frame):
    img = cv2.resize(frame, (320, 320))
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    img = (img.astype(np.float32) - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

# ============================================================
# MAIN UDP SOCKET STREAM RECEIVER LOOP
# ============================================================
def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(("0.0.0.0", 9998))
    logger.info("URLLC Two-Way UDP NanoDet Stats-Enabled Server listening on port 9998...")

    buffer = {}  
    last_packet_time = time.time()
    summary_printed = True

    while True:
        try:
            # Set a non-blocking timeout check to detect when a streaming session has ended
            server_socket.settimeout(3.0)
            try:
                packet, addr = server_socket.recvfrom(MAX_PACKET_SIZE)
                server_metrics["packets_received"] += 1
                last_packet_time = time.time()
                summary_printed = False
            except socket.timeout:
                # If 3 seconds pass without data and we haven't printed the stats yet, output the report
                if not summary_printed:
                    print_server_session_summary()
                    summary_printed = True
                continue

            if len(packet) < HEADER_SIZE:
                continue

            header = packet[:HEADER_SIZE]
            frame_idx, chunk_idx, total_chunks, data_len = struct.unpack(HEADER_FORMAT, header)
            chunk_data = packet[HEADER_SIZE:HEADER_SIZE + data_len]

            server_metrics["frames_attempted"].add(frame_idx)

            if frame_idx not in buffer:
                buffer[frame_idx] = {"chunks": {}, "total": total_chunks, "timestamp": time.perf_counter()}

            buffer[frame_idx]["chunks"][chunk_idx] = chunk_data

            # If all fragments for this frame have arrived successfully
            if len(buffer[frame_idx]["chunks"]) == total_chunks:
                start_pipeline = time.perf_counter()
                
                full_frame_bytes = b"".join([buffer[frame_idx]["chunks"][i] for i in range(total_chunks)])
                np_arr = np.frombuffer(full_frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                detected_objects = []
                if frame is not None:
                    blob = preprocess_matrix(frame)
                    outputs = ort_session.run(None, {input_name: blob})
                    scores_matrix = outputs[0][0][:, :80]
                    class_ids = np.argmax(scores_matrix, axis=-1)
                    max_scores = np.max(scores_matrix, axis=-1)
                    
                    detected = set()
                    for idx, score in enumerate(max_scores):
                        if score > 0.45:
                            class_idx = int(class_ids[idx])
                            if class_idx < len(CLASSES):
                                detected.add(CLASSES[class_idx])
                    detected_objects = list(detected)
                    
                latency_ms = (time.perf_counter() - start_pipeline) * 1000
                print(f"[NANODET FRAME {frame_idx:03d}] Processing Complete: {latency_ms:.2f}ms")
                
                # Update server metrics trackers
                server_metrics["frames_reassembled"] += 1
                server_metrics["total_inference_time"] += latency_ms
                
                # Return telemetry
                telemetry_payload = json.dumps({
                    "frame_idx": frame_idx,
                    "inference_time_ms": round(latency_ms, 2),
                    "detected": detected_objects
                }).encode('utf-8')
                
                server_socket.sendto(telemetry_payload, addr)
                del buffer[frame_idx]

            # Periodic garbage collection for fragments broken by wireless packet drops
            current_time = time.perf_counter()
            old_frames = [fid for fid, f_data in buffer.items() if current_time - f_data["timestamp"] > 1.0]
            for fid in old_frames:
                server_metrics["frames_dropped"] += 1
                del buffer[fid]
                
        except Exception as e:
            print(f"Server Error: {e}")

if __name__ == "__main__":
    main()