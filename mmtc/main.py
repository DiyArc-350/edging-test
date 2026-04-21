import json
import os
import sys
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
DATA_FILE = "sensor_storage.json"

# Setup Logging so it displays clearly in the Docker logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger('webserver')

# Load existing data if it exists
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        stored_data = json.load(f)
else:
    stored_data = {}

@app.route('/collect', methods=['POST'])
def collect():
    data = request.json
    device_id = data['device_id']
    
    # Show log in Docker
    logger.info(f"Data Collected -> Device: {device_id} | Payload size: {data.get('size_kb', 0)} KB")

    # Save/Update the data for this specific UE
    stored_data[device_id] = {
        "data": data['data'][:50] + "...", # Store a preview
        "full_size": data.get('size_kb'),
        "timestamp": data.get('timestamp')
    }

    with open(DATA_FILE, 'w') as f:
        json.dump(stored_data, f)

    return {"status": "saved"}, 201

# NEW: Allow UEs to retrieve their last saved data
@app.route('/get_data/<device_id>', methods=['GET'])
def get_data(device_id):
    device_record = stored_data.get(device_id)
    if device_record:
        logger.info(f"Data Served -> Device: {device_id}")
        return jsonify(device_record), 200
        
    logger.warning(f"Data Not Found for -> Device: {device_id}")
    return {"error": "No data found"}, 404

if __name__ == '__main__':
    logger.info("Initializing Simple Flask Logger on port 8080...")
    app.run(host='0.0.0.0', port=8080)
