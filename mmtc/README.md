# 5G MMTC Webserver

This directory contains a simple Flask webserver designed to collect sensor data from devices, simulating a 5G MMTC (Massive Machine-Type Communications) environment.

## Prerequisites

- Docker
- Docker Compose

## How to Run

1. Open your terminal and navigate to this directory (`mmtc`).
2. Build and start the container in detached mode by running:
   ```bash
   docker-compose up -d --build
   ```
3. The server will start and be accessible at `http://localhost:8080`.

## API Endpoints

- **POST** `/mmtc/collect`
  - Accepts JSON payload with `device_id`, `size_kb`, `timestamp`, and `data`.
  - Saves the data to `sensor_storage.json`.

- **GET** `/mmtc/get_data/<device_id>`
  - Retrieves the last saved data for the specified device ID.

## How to Stop

To stop the webserver, run:
```bash
docker-compose down
```
