from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import jwt
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
import time
import secrets
import uvicorn
import logging
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import firestore as google_firestore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Configuration & Environment -----
ML_SHARED_SECRET = os.getenv("ML_SHARED_SECRET", "demo-internal-secret")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "demo-jwt-secret-key-for-development")
ALGORITHM = "HS256"

# ----- FastAPI App -----
app = FastAPI(
    title="Dubionic Monitoring API",
    description="Secure hydroponic system monitoring with ML predictions",
    version="1.0.0"
)

# CORS
allowed = os.getenv("ALLOWED_ORIGINS", "")
if allowed:
    allow_list = [o.strip() for o in allowed.split(",") if o.strip()]
else:
    allow_list = ["*"]  # default for dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting
rate_limit_cache: Dict[str, List[float]] = {}
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 100

# Globals
monitor = None
db = None
executor = ThreadPoolExecutor(max_workers=2)

# Cache
cache: Dict[str, Any] = {}
CACHE_TTL = 30000

# ----- Data models -----
class SensorData(BaseModel):
    userId: Optional[str] = None
    u: Optional[str] = None
    ph: Optional[float] = None
    p: Optional[float] = None
    ec: Optional[float] = None
    e: Optional[float] = None
    waterTemp: Optional[float] = None
    wt: Optional[float] = None
    airTemp: Optional[float] = None
    at: Optional[float] = None
    humidity: Optional[float] = None
    h: Optional[float] = None
    doorStatus: Optional[str] = None
    ds: Optional[str] = None
    d: Optional[str] = None
    pumpRunning: Optional[bool] = None
    pr: Optional[bool] = None

class PredictionRequest(BaseModel):
    sensor_data: Optional[Dict[str, Any]] = None

class LoginRequest(BaseModel):
    user_id: str

# ----- Firebase Initialization -----
def initialize_firebase():
    try:
        # Check if Firebase app is already initialized
        firebase_admin.get_app()
        return firestore.client()
    except ValueError:
        # Initialize with environment variables (for Replit/cloud deployment)
        if all([os.getenv('FIREBASE_PROJECT_ID'), os.getenv('FIREBASE_CLIENT_EMAIL'), os.getenv('FIREBASE_PRIVATE_KEY')]):
            cred_dict = {
                "type": "service_account",
                "project_id": os.getenv('FIREBASE_PROJECT_ID'),
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),
                "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
                "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
                "client_id": os.getenv('FIREBASE_CLIENT_ID', ''),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL', '')
            }
            cred = credentials.Certificate(cred_dict)
        else:
            # For local development with service account file
            cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', './smart-hydroponic-7d894-firebase-adminsdk-fbsvc-bdb52cbdaf.json')
            if not os.path.exists(cred_path):
                logger.warning("Firebase credentials not found. Using mock data mode.")
                return None
            cred = credentials.Certificate(cred_path)

        firebase_admin.initialize_app(cred)
        return firestore.client()

# ----- Helpers -----
def create_jwt_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None
):
    # Check internal token first (for server-to-server)
    if request:
        internal_token = request.headers.get("x-internal-token")
        if internal_token and internal_token == ML_SHARED_SECRET:
            uid = request.headers.get("x-user-id")
            if not uid:
                raise HTTPException(status_code=400, detail="x-user-id header required for internal requests")
            return uid

    # Fall back to JWT auth
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials
    payload = verify_jwt_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    return user_id

def rate_limit_check(client_ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    rate_limit_cache.setdefault(client_ip, [])
    rate_limit_cache[client_ip] = [t for t in rate_limit_cache[client_ip] if t > window_start]
    if len(rate_limit_cache[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rate_limit_cache[client_ip].append(now)


# Time helpers
def get_colombo_iso():
    """Return ISO 8601 timestamp in Asia/Colombo timezone.

    Tries zoneinfo, falls back to pytz if available, otherwise uses a fixed +05:30 offset.
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Colombo")).isoformat()
    except Exception:
        try:
            import pytz
            return datetime.now(pytz.timezone("Asia/Colombo")).isoformat()
        except Exception:
            # Fallback to fixed offset +05:30
            return datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()

# ----- Monitor class with Firebase Integration -----
class DubionicMonitor:
    def __init__(self, firestore_db):
        self.health_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.sensor_history = []
        self.db = firestore_db
        logger.info("DubionicMonitor initialized with Firebase")

    def get_latest_sensor_data(self, user_id):
        """Get latest sensor data from Firebase - matches Node.js server behavior"""
        try:
            # Check cache first
            cache_key = f"latest_{user_id}"
            cached = cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < CACHE_TTL/1000:
                return cached['data']

            if not self.db:
                # Fallback to mock data if Firebase not available
                return self._get_mock_sensor_data(user_id)

            # Query Firebase like Node.js server
            docs = list(self.db.collection("sensorData")
                       .where("userId", "==", user_id)
                       .order_by("timestamp", direction=firestore.Query.DESCENDING)
                       .limit(1)
                       .stream())

            if not docs:
                logger.info(f"No sensor data found for user {user_id}")
                return None

            data = docs[0].to_dict()

            # Convert Firestore timestamp to Python datetime
            if 'timestamp' in data and hasattr(data['timestamp'], 'to_pydatetime'):
                data['timestamp'] = data['timestamp'].to_pydatetime()

            # Ensure a Colombo-local ISO timestamp is present for display
            if 'timestamp_colombo' not in data:
                try:
                    # Convert the timestamp (aware or naive) to Asia/Colombo
                    ts = data.get('timestamp')
                    if ts is None:
                        data['timestamp_colombo'] = get_colombo_iso()
                    else:
                        try:
                            from zoneinfo import ZoneInfo
                            # If ts is naive, assume it's UTC
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            data['timestamp_colombo'] = ts.astimezone(ZoneInfo('Asia/Colombo')).isoformat()
                        except Exception:
                            try:
                                import pytz
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                                data['timestamp_colombo'] = ts.astimezone(pytz.timezone('Asia/Colombo')).isoformat()
                            except Exception:
                                # fallback
                                data['timestamp_colombo'] = get_colombo_iso()
                except Exception:
                    data['timestamp_colombo'] = get_colombo_iso()

            # Cache the result
            cache[cache_key] = {'data': data, 'timestamp': time.time()}

            logger.info(f"Retrieved sensor data for user {user_id}")
            return data

        except Exception as e:
            logger.error(f"Error fetching sensor data from Firebase: {e}")
            # Fallback to mock data
            return self._get_mock_sensor_data(user_id)

    def _get_mock_sensor_data(self, user_id):
        """Fallback mock data when Firebase is unavailable"""
        mock_data = {
            'ph': round(5.5 + np.random.random() * 1.5, 2),
            'ec': round(1.0 + np.random.random() * 1.5, 2),
            'waterTemp': round(18 + np.random.random() * 10, 1),
            'airTemp': round(20 + np.random.random() * 8, 1),
            'humidity': round(60 + np.random.random() * 20, 1),
            'userId': user_id,
            'timestamp': get_colombo_iso()
        }
        logger.info(f"Using mock sensor data for user {user_id}")
        return mock_data

    def save_sensor_data(self, sensor_data):
        """Save sensor data to Firebase - matches Node.js server behavior"""
        try:
            # Always attach Colombo local timestamp for client-side display/audit
            sensor_data_colombo_ts = get_colombo_iso()

            if not self.db:
                # Fallback: write to local JSONL when Firebase unavailable
                try:
                    out_path = os.getenv('SENSOR_JSONL_PATH', 'sensor_data.jsonl')
                    parent = os.path.dirname(out_path)
                    if parent and not os.path.exists(parent):
                        os.makedirs(parent, exist_ok=True)

                    processed_local = {
                        'ph': sensor_data.get('ph') or sensor_data.get('p'),
                        'ec': sensor_data.get('ec') or sensor_data.get('e'),
                        'waterTemp': sensor_data.get('waterTemp') or sensor_data.get('wt'),
                        'airTemp': sensor_data.get('airTemp') or sensor_data.get('at'),
                        'humidity': sensor_data.get('humidity') or sensor_data.get('h'),
                        'doorStatus': sensor_data.get('doorStatus') or sensor_data.get('ds') or sensor_data.get('d') or 'unknown',
                        'pumpRunning': sensor_data.get('pumpRunning') if sensor_data.get('pumpRunning') is not None else sensor_data.get('pr'),
                        'userId': sensor_data.get('userId') or sensor_data.get('u'),
                        'timestamp': datetime.utcnow().isoformat(),
                        'timestamp_colombo': sensor_data_colombo_ts,
                        'status': 'active'
                    }
                    # Remove None values
                    processed_local = {k: v for k, v in processed_local.items() if v is not None}
                    with open(out_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(processed_local, ensure_ascii=False) + '\n')
                    logger.info(f"Saved sensor data to local JSONL fallback: {out_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to write sensor data to local JSONL: {e}")
                    return False

            # Process data like Node.js server
            final_user_id = sensor_data.get('userId') or sensor_data.get('u')
            if not final_user_id:
                logger.error("No user ID provided in sensor data")
                return False

            # Handle doorStatus with fallbacks like Node.js
            door_status = sensor_data.get('doorStatus')
            if door_status is None:
                door_status = sensor_data.get('ds')
            if door_status is None:
                door_status = sensor_data.get('d')
            if door_status is None:
                door_status = "unknown"

            # Handle pumpRunning with fallbacks like Node.js
            pump_running = sensor_data.get('pumpRunning')
            if pump_running is None:
                pump_running = sensor_data.get('pr')

            # Build the sensor data document like Node.js
            processed_data = {
                'ph': sensor_data.get('ph') or sensor_data.get('p'),
                'ec': sensor_data.get('ec') or sensor_data.get('e'),
                'waterTemp': sensor_data.get('waterTemp') or sensor_data.get('wt'),
                'airTemp': sensor_data.get('airTemp') or sensor_data.get('at'),
                'humidity': sensor_data.get('humidity') or sensor_data.get('h'),
                'doorStatus': door_status,
                'pumpRunning': pump_running,
                'userId': final_user_id,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'timestamp_colombo': sensor_data_colombo_ts,
                'status': 'active'
            }

            # Remove None values
            processed_data = {k: v for k, v in processed_data.items() if v is not None}

            # Save to sensorData collection
            self.db.collection("sensorData").add(processed_data)

            # Update systemStatus like Node.js
            status_update = {
                'status': 'active',
                'lastUpdated': firestore.SERVER_TIMESTAMP,
                'userId': final_user_id
            }

            if processed_data.get('doorStatus'):
                status_update['doorStatus'] = processed_data['doorStatus']
            if processed_data.get('pumpRunning') is not None:
                status_update['waterPump'] = 'running' if processed_data['pumpRunning'] else 'stopped'

            self.db.collection("systemStatus").document(final_user_id).set(status_update, merge=True)

            # Clear cache
            cache_key = f"latest_{final_user_id}"
            if cache_key in cache:
                del cache[cache_key]

            logger.info(f"Sensor data saved successfully for user {final_user_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving sensor data to Firebase: {e}")
            return False

    def create_training_data_optimized(self, user_id, limit=500):
        """Create training data from historical Firebase data"""
        try:
            if not self.db:
                logger.info("Firebase not available, using mock training data")
                return self._create_fallback_data()

            # Get historical data from Firebase
            docs = list(self.db.collection("sensorData")
                       .where("userId", "==", user_id)
                       .order_by("timestamp", direction=firestore.Query.DESCENDING)
                       .limit(limit)
                       .stream())

            historical_data = []
            for doc in docs:
                data = doc.to_dict()
                # Extract relevant fields for training
                reading = {
                    'temperature': data.get('airTemp'),
                    'humidity': data.get('humidity'),
                    'water_temp': data.get('waterTemp'),
                    'ph': data.get('ph'),
                    'ec': data.get('ec'),
                }
                # Only include complete readings
                if all(v is not None for v in reading.values()):
                    historical_data.append(reading)

            if len(historical_data) < 10:
                logger.warning(f"Insufficient historical data ({len(historical_data)} records). Using fallback data.")
                return self._create_fallback_data()

            df = pd.DataFrame(historical_data)

            # Create health status labels based on parameter thresholds
            conditions = np.select([
                (df['ph'] < 5.5) | (df['ph'] > 6.8),
                (df['ec'] < 1.0) | (df['ec'] > 2.5),
                (df['temperature'] < 18) | (df['temperature'] > 30),
                (df['water_temp'] < 18) | (df['water_temp'] > 25),
                (df['humidity'] < 50) | (df['humidity'] > 85),
            ], [
                'needs_ph_adjust',
                'needs_nutrient_adjust',
                'needs_temp_adjust',
                'needs_water_temp_adjust',
                'needs_humidity_adjust'
            ], default='healthy')

            df['health_status'] = conditions
            df = df.dropna()

            logger.info(f"Created training data with {len(df)} samples from Firebase")
            return df

        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return self._create_fallback_data()

    def _create_fallback_data(self):
        """Create fallback training data when Firebase is unavailable"""
        np.random.seed(42)
        n_samples = 500

        sample_data = {
            'temperature': np.random.normal(24, 3, n_samples),
            'humidity': np.random.normal(70, 10, n_samples),
            'water_temp': np.random.normal(22, 2, n_samples),
            'ph': np.random.normal(6.0, 0.5, n_samples),
            'ec': np.random.normal(1.5, 0.4, n_samples),
        }

        df = pd.DataFrame(sample_data)
        conditions = np.select([
            (df['ph'] < 5.5) | (df['ph'] > 6.8),
            (df['ec'] < 1.0) | (df['ec'] > 2.5),
            (df['temperature'] < 18) | (df['temperature'] > 30),
            (df['water_temp'] < 18) | (df['water_temp'] > 25),
            (df['humidity'] < 50) | (df['humidity'] > 85),
        ], [
            'needs_ph_adjust',
            'needs_nutrient_adjust',
            'needs_temp_adjust',
            'needs_water_temp_adjust',
            'needs_humidity_adjust'
        ], default='healthy')

        df['health_status'] = conditions
        logger.info("Using fallback training data")
        return df

    def train_models_optimized(self, user_id):
        try:
            logger.info(f"Starting model training for user {user_id}")
            df = self.create_training_data_optimized(user_id)

            features = ['temperature', 'humidity', 'water_temp', 'ph', 'ec']
            X = df[features].values
            y_health = df['health_status'].values

            X_train, X_test, y_health_train, y_health_test = train_test_split(
                X, y_health, test_size=0.2, random_state=42, stratify=y_health
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train classifier
            self.health_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.health_classifier.fit(X_train_scaled, y_health_train)
            self.is_trained = True

            # Calculate accuracy
            accuracy = accuracy_score(y_health_test, self.health_classifier.predict(X_test_scaled))

            logger.info(f"Model training completed for {user_id}. Accuracy: {accuracy:.2%}")
            return accuracy

        except Exception as e:
            logger.error(f"Training failed for {user_id}: {e}")
            raise

    def predict_plant_status(self, temperature, humidity, water_temp, ph, ec):
        if not self.is_trained:
            raise Exception("Models not trained. Please train first.")

        features = np.array([[temperature, humidity, water_temp, ph, ec]])
        features_scaled = self.scaler.transform(features)
        health_status = self.health_classifier.predict(features_scaled)[0]

        prediction = {
            'timestamp': datetime.now(),
            'temperature': temperature,
            'humidity': humidity,
            'water_temp': water_temp,
            'ph': ph,
            'ec': ec,
            'health_status': health_status,
        }

        logger.info(f"Prediction made: {health_status}")
        return prediction

    def get_recommendations(self, prediction):
        recommendations = []
        ph = prediction['ph']
        ec = prediction['ec']
        temp = prediction['temperature']
        humidity = prediction['humidity']
        water_temp = prediction['water_temp']

        # Health status based recommendations
        if prediction['health_status'] != 'healthy':
            status_msg = prediction['health_status'].replace('_', ' ').title()
            recommendations.append(f"System Alert: {status_msg}")

        # Specific parameter recommendations
        if ph < 5.5:
            recommendations.append(f"Low pH ({ph:.1f}): Add pH UP solution")
        elif ph > 6.8:
            recommendations.append(f"High pH ({ph:.1f}): Add pH DOWN solution")

        if ec < 1.0:
            recommendations.append(f"Low EC ({ec:.1f}): Increase nutrient concentration")
        elif ec > 2.5:
            recommendations.append(f"High EC ({ec:.1f}): Dilute with fresh water")

        if temp < 18:
            recommendations.append(f"Low air temperature ({temp:.1f}°C): Increase heating")
        elif temp > 30:
            recommendations.append(f"High air temperature ({temp:.1f}°C): Improve ventilation")

        if water_temp < 18:
            recommendations.append(f"Low water temperature ({water_temp:.1f}°C): Use water heater")
        elif water_temp > 25:
            recommendations.append(f"High water temperature ({water_temp:.1f}°C): Cool reservoir")

        if humidity < 50:
            recommendations.append(f"Low humidity ({humidity:.1f}%): Increase misting")
        elif humidity > 85:
            recommendations.append(f"High humidity ({humidity:.1f}%): Improve air circulation")

        if not recommendations:
            recommendations.extend([
                "All parameters are optimal",
                "Maintain current nutrient levels",
                "Continue regular monitoring"
            ])

        return recommendations[:5]

# ----- Startup event -----
@app.on_event("startup")
async def startup_event():
    global monitor, db
    try:
        db = initialize_firebase()
        monitor = DubionicMonitor(db)
        logger.info("Dubionic Monitor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")

# ----- Middleware -----
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"Request completed: {response.status_code} in {process_time:.2f}s")

    return response

# ----- Routes -----
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Dubionic Monitoring API",
        "version": "1.0.0",
        "firebase_connected": db is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "models_trained": monitor.is_trained if monitor else False,
        "service": "operational",
        "firebase_connected": db is not None
    }

@app.get("/api/train-status")
async def api_train_status(request: Request):
    internal_token = request.headers.get("x-internal-token")
    user_id = request.headers.get("x-user-id")

    if internal_token and internal_token == ML_SHARED_SECRET:
        if not user_id:
            return JSONResponse(
                {"detail": "x-user-id header required for internal train-status"},
                status_code=400
            )
        return {
            'is_trained': monitor.is_trained if monitor else False,
            'last_trained': None,
            'accuracy': None,
            'user_id': user_id
        }

    if not monitor:
        return {'is_trained': False, 'last_trained': None, 'accuracy': None}

    return {
        'is_trained': bool(monitor.is_trained),
        'last_trained': None,
        'accuracy': None
    }

@app.post("/auth/login")
async def login(login_data: LoginRequest):
    user_id = login_data.user_id

    if not user_id or len(user_id) < 3:
        raise HTTPException(status_code=400, detail="Valid user ID required (min 3 characters)")
    if len(user_id) > 100:
        raise HTTPException(status_code=400, detail="User ID too long")

    token = create_jwt_token({"sub": user_id})

    logger.info(f"Login successful for user: {user_id}")

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "expires_in": 24 * 60 * 60  # 24 hours in seconds
    }

# Alternative login endpoint with query parameter for convenience
@app.post("/auth/login-simple")
async def login_simple(user_id: str):
    if not user_id or len(user_id) < 3:
        raise HTTPException(status_code=400, detail="Valid user ID required")

    token = create_jwt_token({"sub": user_id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id
    }

@app.get("/api/sensor-data/latest")
async def get_latest_sensor_data(
    userId: str,
    current_user: str = Depends(get_current_user)
):
    if not monitor:
        raise HTTPException(status_code=503, detail="Service unavailable")

    if userId != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    data = monitor.get_latest_sensor_data(userId)
    if not data:
        raise HTTPException(status_code=404, detail="No data found")

    return data

@app.post("/api/sensor-data")
async def receive_sensor_data(
    sensor_data: SensorData,
    background_tasks: BackgroundTasks
):
    """Receive sensor data - matches Node.js server behavior exactly"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Service unavailable")

    # Return immediate response like Node.js server
    response = {
        "message": "Data received",
        "received": True,
        "timestamp": datetime.now().isoformat()
    }

    # Process data in background like Node.js setTimeout
    def process_data():
        try:
            success = monitor.save_sensor_data(sensor_data.dict())
            if success:
                logger.info("Sensor data processed and saved successfully")
            else:
                logger.error("Failed to process sensor data")
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")

    # Add small delay to mimic Node.js setTimeout behavior
    background_tasks.add_task(lambda: time.sleep(0.1) or process_data())

    return response


@app.get('/admin/local-sensor-latest')
async def admin_local_sensor_latest():
    """Return the last line from the local sensor_data.jsonl fallback for debugging.

    Useful when Firestore is not connected on PythonAnywhere or local dev.
    """
    out_path = os.getenv('SENSOR_JSONL_PATH', 'sensor_data.jsonl')
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Local sensor JSONL not found")

    try:
        # Read last non-empty line efficiently
        last_line = None
        with open(out_path, 'rb') as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            for line in f:
                if line.strip():
                    last_line = line.decode('utf-8')

        if not last_line:
            raise HTTPException(status_code=404, detail="No entries in local sensor JSONL")

        return JSONResponse(status_code=200, content=json.loads(last_line))
    except Exception as e:
        logger.error(f"Error reading local sensor JSONL: {e}")
        raise HTTPException(status_code=500, detail="Failed to read local sensor JSONL")

@app.post("/train")
async def train_models(
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    if not monitor:
        raise HTTPException(status_code=503, detail="Service unavailable")

    logger.info(f"Training request received for user: {current_user}")

    def train_task():
        try:
            accuracy = monitor.train_models_optimized(current_user)
            logger.info(f"Background training completed for {current_user}")
            return accuracy
        except Exception as e:
            logger.error(f"Background training failed for {current_user}: {e}")
            return 0.0

    background_tasks.add_task(train_task)

    return {
        "message": "Training started in background",
        "user_id": current_user,
        "status": "processing",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def make_prediction(
    request_data: PredictionRequest,
    current_user: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = None
):
    if not monitor:
        raise HTTPException(status_code=503, detail="Service unavailable")

    if not monitor.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Models not trained yet. Please train first by calling /train endpoint."
        )

    try:
        logger.info(f"Prediction request from user: {current_user}")

        if request_data.sensor_data:
            sensor = request_data.sensor_data
            prediction = monitor.predict_plant_status(
                temperature=sensor.get('temperature', sensor.get('airTemp', 24)),
                humidity=sensor.get('humidity', 70),
                water_temp=sensor.get('waterTemp', 21),
                ph=sensor.get('ph', 6.0),
                ec=sensor.get('ec', 1.5)
            )
        else:
            # Use latest sensor data from Firebase
            sensor_data = monitor.get_latest_sensor_data(current_user)
            if not sensor_data:
                raise HTTPException(status_code=404, detail="No sensor data found for prediction")

            prediction = monitor.predict_plant_status(
                temperature=sensor_data.get('airTemp', 24),
                humidity=sensor_data.get('humidity', 70),
                water_temp=sensor_data.get('waterTemp', 21),
                ph=sensor_data.get('ph', 6.0),
                ec=sensor_data.get('ec', 1.5)
            )

        # Get recommendations
        recommendations = monitor.get_recommendations(prediction)

        result = {
            "prediction_id": f"pred_{int(time.time())}",
            "health_status": prediction['health_status'],
            "recommendations": recommendations,
            "critical": any('Alert' in rec for rec in recommendations),
            "timestamp": prediction['timestamp'].isoformat(),
            "sensor_data": {
                'temperature': prediction['temperature'],
                'humidity': prediction['humidity'],
                'water_temp': prediction['water_temp'],
                'ph': prediction['ph'],
                'ec': prediction['ec']
            },
            "user_id": current_user
        }

        # Persist prediction: Firestore if available, otherwise local JSONL fallback
        try:
            # Add Colombo local timestamp alongside server timestamp
            colombo_ts = get_colombo_iso()
            doc = {
                'prediction_id': result['prediction_id'],
                'user_id': current_user,
                'health_status': result['health_status'],
                'recommendations': recommendations,
                'critical': result['critical'],
                # store server timestamp for Firestore; fallback uses ISO string
                'timestamp': firestore.SERVER_TIMESTAMP if (monitor and getattr(monitor, 'db', None)) else datetime.utcnow().isoformat(),
                'timestamp_colombo': colombo_ts,
                'sensor_data': result['sensor_data']
            }

            if monitor and getattr(monitor, 'db', None):
                try:
                    # Use prediction_id as document id for idempotency
                    monitor.db.collection('predictions').document(result['prediction_id']).set(doc)
                    logger.info(f"Saved prediction {result['prediction_id']} to Firestore for user {current_user}")
                except Exception as e:
                    logger.error(f"Failed to save prediction to Firestore: {e}")
            else:
                # Local JSONL fallback
                try:
                    out_path = os.getenv('PREDICTIONS_JSONL_PATH', 'predictions.jsonl')
                    parent = os.path.dirname(out_path)
                    if parent and not os.path.exists(parent):
                        os.makedirs(parent, exist_ok=True)
                    # Ensure timestamp is serializable
                    local_doc = dict(doc)
                    if isinstance(local_doc.get('timestamp'), datetime):
                        local_doc['timestamp'] = local_doc['timestamp'].isoformat()
                    with open(out_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(local_doc, ensure_ascii=False) + '\n')
                    logger.info(f"Saved prediction to local JSONL fallback: {out_path}")
                except Exception as e:
                    logger.error(f"Failed to write prediction to local JSONL: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while persisting prediction: {e}")

        logger.info(f"Prediction successful for {current_user}: {prediction['health_status']}")
        return result

    except Exception as e:
        logger.error(f"Prediction failed for {current_user}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/ping")
async def ping():
    return {
        "status": "OK",
        "message": "Server is running",
        "timestamp": datetime.now().isoformat(),
        "firebase_connected": db is not None
    }

# Public endpoint to verify tokens
@app.get("/auth/verify")
async def verify_token(current_user: str = Depends(get_current_user)):
    return {
        "valid": True,
        "user_id": current_user,
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    def get_local_ip():
        """Attempt to determine the machine's local IP address for display.

        This does not change the bind address (we still bind to 0.0.0.0),
        but prints a convenient URL that other devices on the LAN can use.
        """
        try:
            import socket

            # Connect to an external address (doesn't send data) to get the
            # preferred outbound IP for this host. Works without internet if
            # there's a network route (uses a non-routable address).
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            try:
                # This address is arbitrary and won't be contacted.
                s.connect(("198.51.100.1", 80))
                ip = s.getsockname()[0]
            except Exception:
                # Fallback to localhost
                ip = "127.0.0.1"
            finally:
                s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()
    logger.info(f"Server will listen on 0.0.0.0:{port} (accessible on LAN at http://{local_ip}:{port})")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        loop="asyncio",
        access_log=True,
        log_level="info"
    )
