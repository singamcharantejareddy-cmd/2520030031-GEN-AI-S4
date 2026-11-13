from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import PyMongoError


class MongoLogger:

    def __init__(self):
        self.client = None
        self.collection = None

        try:
            self.client = MongoClient(
                "mongodb://localhost:27017/",
                serverSelectionTimeoutMS=2000
            )

            self.client.admin.command("ping")

            db = self.client["llm_monitoring"]
            self.collection = db["call_logs"]

            print("MongoDB connected successfully!")

        except PyMongoError as error:
            print("MongoDB connection failed:", error)

    def log_call(self, record):

        if self.collection is None:
            return False

        record["timestamp"] = datetime.now(timezone.utc)

        try:
            self.collection.insert_one(record)
            return True

        except PyMongoError as error:
            print("Logging error:", error)
            return False

    def close(self):

        if self.client:
            self.client.close()