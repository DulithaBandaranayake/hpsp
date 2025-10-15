from flask import Flask, request, jsonify
import os
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None


def initialize_firebase():
    if not firebase_admin:
        logger.warning("firebase_admin not available; running in mock mode")
        return None

    try:
        # If explicit service account pieces are provided via env vars use them
        if all([os.getenv('FIREBASE_PROJECT_ID'), os.getenv('FIREBASE_CLIENT_EMAIL'), os.getenv('FIREBASE_PRIVATE_KEY')]):
            cred_dict = {
                "type": "service_account",
                "project_id": os.getenv('FIREBASE_PROJECT_ID'),
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),
                "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
                "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
                "client_id": os.getenv('FIREBASE_CLIENT_ID', ''),
            }
            cred = credentials.Certificate(cred_dict)
        else:
            cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', './smart-hydroponic-7d894-firebase-adminsdk-fbsvc-bdb52cbdaf.json')
            if not os.path.exists(cred_path):
                logger.warning("Firebase credentials not found; running in mock mode")
                return None
            cred = credentials.Certificate(cred_path)

        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        logger.exception("Failed to initialize Firebase: %s", e)
        return None


def save_sensor_data_to_firestore(db, sensor_data: dict) -> bool:
    try:
        if not db:
            # JSONL fallback for environments without outbound access (e.g. PythonAnywhere free tier)
            try:
                out_path = os.getenv('SENSOR_JSONL_PATH', 'sensor_data.jsonl')
                entry = dict(sensor_data)
                entry['_received_at'] = time.time()
                # ensure directory exists for explicit paths
                parent = os.path.dirname(out_path)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                logger.info("Saved sensor payload to JSONL fallback: %s", out_path)
                return True
            except Exception:
                logger.exception("Failed to write sensor payload to JSONL fallback")
                return False

        final_user_id = sensor_data.get('userId') or sensor_data.get('u')
        if not final_user_id:
            logger.error("No user id in sensor payload")
            return False

        door_status = sensor_data.get('doorStatus')
        if door_status is None:
            door_status = sensor_data.get('ds')
        if door_status is None:
            door_status = sensor_data.get('d')
        if door_status is None:
            door_status = 'unknown'

        pump_running = sensor_data.get('pumpRunning')
        if pump_running is None:
            pump_running = sensor_data.get('pr')

        processed = {
            'ph': sensor_data.get('ph') or sensor_data.get('p'),
            'ec': sensor_data.get('ec') or sensor_data.get('e'),
            'waterTemp': sensor_data.get('waterTemp') or sensor_data.get('wt'),
            'airTemp': sensor_data.get('airTemp') or sensor_data.get('at'),
            'humidity': sensor_data.get('humidity') or sensor_data.get('h'),
            'doorStatus': door_status,
            'pumpRunning': pump_running,
            'userId': final_user_id,
            'timestamp': firestore.SERVER_TIMESTAMP if db else None,
            'status': 'active'
        }

        # Remove None values
        processed = {k: v for k, v in processed.items() if v is not None}

        db.collection('sensorData').add(processed)

        status_update = {
            'status': 'active',
            'lastUpdated': firestore.SERVER_TIMESTAMP,
            'userId': final_user_id
        }
        if processed.get('doorStatus'):
            status_update['doorStatus'] = processed['doorStatus']
        if processed.get('pumpRunning') is not None:
            status_update['waterPump'] = 'running' if processed['pumpRunning'] else 'stopped'

        db.collection('systemStatus').document(final_user_id).set(status_update, merge=True)

        return True
    except Exception as e:
        logger.exception("Error saving sensor data: %s", e)
        return False


app = Flask(__name__)
db = initialize_firebase()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'OK',
        'firebase_connected': db is not None
    })


@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    if not request.is_json:
        return jsonify({'detail': 'JSON required'}), 400
    payload = request.get_json()
    # Quick basic validation
    if not isinstance(payload, dict):
        return jsonify({'detail': 'Invalid payload'}), 400

    # Save to Firestore (or mock)
    saved = save_sensor_data_to_firestore(db, payload)

    return jsonify({
        'message': 'Data received',
        'received': True,
        'saved': bool(saved)
    })
