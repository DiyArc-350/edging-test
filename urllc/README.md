# YOLOv8 Deployment API

This directory contains a FastAPI backend service that uses YOLOv8 to detect objects in uploaded videos. The service is containerized using Docker.

## Prerequisites

- Docker
- Docker Compose

## How to Run

1. Open your terminal and navigate to this directory (`yolo-deploy`).
2. Build and start the Docker container in detached mode by running:
   ```bash
   docker-compose up -d --build
   ```
3. The API will be accessible at `http://localhost:8001` (Note: The host port is mapped to 8001, while the container internally uses 8000).

## API Endpoints

- **POST** `/upload/`
  - Accepts a video file (`.mp4`, `.avi`, `.mov`) via form data (`file`).
  - Returns a JSON response containing the `upload_time`, `inference_time` (YOLO processing time), and a list of `detected` objects in the first few frames of the video.

## Notes

- The Docker container uses a volume named `yolo_models` to cache the downloaded YOLOv8 weights so it doesn't have to download them every time it restarts.
- Uploaded videos are temporarily saved in `backend/received_videos/` and deleted after processing.

## How to Stop

To stop the API service, run:
```bash
docker-compose down
```
