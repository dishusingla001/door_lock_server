"""
Cloud Server for ESP32-CAM Door Lock System
Railway-ready | MongoDB Atlas only | No local storage
"""

# only for testing 
# --------------------------------------

# from dotenv import load_dotenv
# load_dotenv()

# ---------------------------------
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import hashlib
import os
from datetime import datetime
import face_recognition
from pyzbar.pyzbar import decode
from mongo_config import MongoDBConfig

# ==================== FLASK APP ====================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ==================== ENV VARIABLES ====================

QR_HASH = os.environ.get("QR_HASH")
if not QR_HASH:
    raise RuntimeError("QR_HASH environment variable not set")
PORT = int(os.environ.get("PORT", 5000))

# ==================== GLOBALS ====================

mongo_db = None
known_face_encodings = []
known_face_names = []
encodings_loaded = False
session_cache = {}

# ==================== INIT ====================

def initialize_system():
    global mongo_db
    print("=" * 50)
    print("ESP32-CAM CLOUD SERVER STARTING")
    print("=" * 50)

    try:
        mongo_uri = os.environ.get("MONGO_URI")
        if not mongo_uri:
            raise Exception("MONGO_URI not set")

        mongo_db = MongoDBConfig(mongo_uri)
        print("✓ MongoDB Atlas connected")

        load_faces_from_mongo()

    except Exception as e:
        print(f"✗ Startup error: {e}")

    print("✓ Server Ready")
    print("=" * 50)


def load_faces_from_mongo():
    global known_face_encodings, known_face_names, encodings_loaded

    encodings, names = mongo_db.get_all_face_encodings()

    if len(encodings) == 0:
        print("⚠ No face encodings found in MongoDB")
        return

    known_face_encodings = encodings
    known_face_names = names
    encodings_loaded = True

    print(f"✓ Loaded {len(encodings)} face encodings")


# ==================== UTILITIES ====================

def decode_qr(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    qr_codes = decode(gray)
    if qr_codes:
        return qr_codes[0].data.decode("utf-8")
    return None


def validate_qr(qr_data):
    if not qr_data:
        return False

    if qr_data == QR_HASH:
        return True

    hashed = hashlib.sha256(qr_data.encode()).hexdigest()
    return hashed == QR_HASH


def recognize_face(image_np):
    if not encodings_loaded:
        return None, "No encodings loaded"

    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")

    if not locations:
        return None, "No face detected"

    encodings = face_recognition.face_encodings(rgb, locations)

    face_encoding = encodings[0]
    distances = face_recognition.face_distance(
        known_face_encodings, face_encoding
    )

    best_match = np.argmin(distances)
    confidence = 1 - distances[best_match]

    if confidence > 0.5:
        return known_face_names[best_match], f"{confidence*100:.1f}%"

    return None, "Face not recognized"


# ==================== ROUTES ====================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "ESP32-CAM Door Lock",
        "status": "running"
    })


@app.route("/api/status", methods=["GET"])
def status():
    print(f"DEBUG: encodings_loaded={encodings_loaded}, known_face_names count={len(known_face_names)}")
    return jsonify({
        "online": True,
        "faces_loaded": encodings_loaded,
        "known_faces": len(set(known_face_names))
    })


@app.route("/api/verify-qr", methods=["POST"])
def verify_qr():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image"}), 400

    image = base64.b64decode(data["image"])
    image_np = cv2.imdecode(
        np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR
    )

    qr_data = decode_qr(image_np)
    valid = validate_qr(qr_data)

    if valid:
        session_id = hashlib.md5(
            str(datetime.now()).encode()
        ).hexdigest()

        session_cache[session_id] = True
        mongo_db.log_access("QR", "opened", "qr")

        return jsonify({
            "valid": True,
            "session_id": session_id
        })

    mongo_db.log_access("QR", "denied", "qr")
    return jsonify({"valid": False})


@app.route("/api/recognize-face", methods=["POST"])
def recognize_face_api():
    if not encodings_loaded:
        return jsonify({
            "error": "No face encodings available"
        }), 503

    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image"}), 400

    image = base64.b64decode(data["image"])
    image_np = cv2.imdecode(
        np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR
    )

    name, msg = recognize_face(image_np)

    if name:
        mongo_db.log_access(name, "opened", "face")
        return jsonify({
            "recognized": True,
            "name": name,
            "confidence": msg,
            "access": "granted"
        })

    mongo_db.log_access("Unknown", "denied", "face")
    return jsonify({
        "recognized": False,
        "access": "denied"
    })


# ==================== MAIN ====================

# Initialize system at module level for gunicorn compatibility
initialize_system()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
