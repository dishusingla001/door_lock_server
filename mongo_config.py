"""
MongoDB Configuration for Face Recognition System
Cloud-ready | MongoDB Atlas | Railway safe
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from datetime import datetime
import numpy as np


class MongoDBConfig:
    def __init__(self, connection_string=None):
        """
        Initialize MongoDB connection

        Args:
            connection_string: MongoDB URI (Atlas or local)
                               If None, reads from MONGO_URI env variable
        """
        if connection_string is None:
            connection_string = os.getenv("MONGO_URI")

        if not connection_string:
            raise Exception("MONGO_URI environment variable not set")

        try:
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000
            )
            self.client.server_info()
            print("✓ Connected to MongoDB successfully")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"✗ MongoDB connection failed: {e}")
            raise

        # Database & collections
        self.db = self.client["face_recognition_db"]
        self.users_collection = self.db["users"]
        self.encodings_collection = self.db["face_encodings"]
        self.access_logs_collection = self.db["access_logs"]

        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for better performance"""
        self.users_collection.create_index("name", unique=True)
        self.encodings_collection.create_index("user_id")
        self.access_logs_collection.create_index("timestamp")
        self.access_logs_collection.create_index("user_name")

    # ==================== USERS ====================

    def add_user(self, name, role="user"):
        try:
            result = self.users_collection.insert_one({
                "name": name,
                "role": role,
                "created_at": datetime.utcnow(),
                "is_active": True
            })
            return str(result.inserted_id)
        except Exception:
            user = self.get_user_by_name(name)
            return str(user["_id"]) if user else None

    def get_user_by_name(self, name):
        return self.users_collection.find_one({"name": name})

    def get_all_users(self):
        return list(self.users_collection.find({"is_active": True}))
    
    # ==================== FACE ENCODINGS ====================

    def save_face_encoding(self, user_name, encoding, image_name=None):
        try:
            user = self.get_user_by_name(user_name)
            user_id = str(user["_id"]) if user else self.add_user(user_name)

            encoding_doc = {
                "user_id": user_id,
                "user_name": user_name,
                "encoding": encoding.tolist(),
                "image_name": image_name,
                "created_at": datetime.utcnow()
            }

            result = self.encodings_collection.insert_one(encoding_doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"✗ Failed to save encoding: {e}")
            return None

    def get_all_face_encodings(self):
        encodings = []
        names = []

        try:
            for doc in self.encodings_collection.find():
                encodings.append(np.array(doc["encoding"]))
                names.append(doc["user_name"])
            return encodings, names
        except Exception as e:
            print(f"✗ Failed to load encodings: {e}")
            return [], []

    def delete_user_encodings(self, user_name):
        try:
            result = self.encodings_collection.delete_many(
                {"user_name": user_name}
            )
            return result.deleted_count
        except Exception:
            return 0

    # ==================== ACCESS LOGS ====================

    def log_access(self, user_name, status, access_type, confidence=None):
        try:
            self.access_logs_collection.insert_one({
                "user_name": user_name,
                "status": status,               # opened / denied
                "access_type": access_type,     # qr / face / manual
                "confidence": confidence,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            print(f"✗ Failed to log access: {e}")

    def get_access_logs(self, limit=100, user_name=None):
        query = {}
        if user_name:
            query["user_name"] = user_name

        return list(
            self.access_logs_collection
            .find(query)
            .sort("timestamp", -1)
            .limit(limit)
        )

    # ==================== CLEANUP ====================

    def close(self):
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")


# ==================== OPTIONAL MIGRATION (LOCAL USE ONLY) ====================

def migrate_local_to_mongodb(dataset_path, mongo):
    """
    Run ONLY on local machine (NOT Railway)
    Migrates dataset images to MongoDB
    """
    import face_recognition
    import os

    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue

        for img in os.listdir(person_dir):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(person_dir, img)
                image = face_recognition.load_image_file(image_path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    mongo.save_face_encoding(person, encs[0], img)
