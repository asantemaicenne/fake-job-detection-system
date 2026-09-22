import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, TEXT

from configs.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Singleton repository managing MongoDB connection pool and indexing strategy."""

    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    async def connect(cls) -> None:
        """Initializes connection pool and ensures all indexes exist."""
        if cls.client is not None:
            logger.warning("Database client already initialized.")
            return

        logger.info(f"Connecting to MongoDB at: {settings.MONGODB_URI.split('@')[-1]}")
        cls.client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            maxPoolSize=settings.MONGODB_MAX_POOL_SIZE,
            minPoolSize=settings.MONGODB_MIN_POOL_SIZE,
            serverSelectionTimeoutMS=5000,
        )
        cls.db = cls.client[settings.MONGODB_DB_NAME]

        # Verify connectivity
        await cls.client.admin.command("ping")
        logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DB_NAME}")

        # Provision performance and TTL indexes
        await cls._create_indexes()

    @classmethod
    async def disconnect(cls) -> None:
        """Closes all active database sockets."""
        if cls.client is not None:
            logger.info("Closing MongoDB connection pool.")
            cls.client.close()
            cls.client = None
            cls.db = None

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """Returns the active database instance."""
        if cls.db is None:
            raise RuntimeError("DatabaseManager has not been connected. Call connect() first.")
        return cls.db

    @classmethod
    def get_collection(cls, collection_name: str) -> AsyncIOMotorCollection:
        """Returns a typed AsyncIOMotorCollection handle."""
        return cls.get_database()[collection_name]

    @classmethod
    async def _create_indexes(cls) -> None:
        """Provisions optimized indexes across all system collections."""
        database = cls.get_database()

        # 1. Indexes for raw_jobs collection
        raw_jobs = database["raw_jobs"]
        await raw_jobs.create_index(
            [("title", TEXT), ("description", TEXT)],
            name="idx_raw_jobs_text_search",
        )
        await raw_jobs.create_index(
            [("source_platform", ASCENDING), ("posted_at", DESCENDING)],
            name="idx_raw_jobs_platform_time",
        )
        await raw_jobs.create_index(
            [("poster_id", ASCENDING), ("posted_at", DESCENDING)],
            name="idx_raw_jobs_poster_history",
        )

        # 2. Indexes for job_features collection
        job_features = database["job_features"]
        await job_features.create_index(
            [("job_id", ASCENDING)],
            unique=True,
            name="idx_features_job_id_unique",
        )

        # 3. Indexes for predictions collection (with 365-day TTL expiry)
        predictions = database["predictions"]
        await predictions.create_index(
            [("job_id", ASCENDING)],
            unique=True,
            name="idx_predictions_job_id_unique",
        )
        await predictions.create_index(
            [("confidence_score", ASCENDING)],
            name="idx_predictions_confidence",
        )
        await predictions.create_index(
            [("timestamp", ASCENDING)],
            expireAfterSeconds=31536000,
            name="idx_predictions_ttl_365d",
        )

        # 4. Indexes for feedback / HITL audit trail
        feedback = database["feedback"]
        await feedback.create_index(
            [("job_id", ASCENDING)],
            unique=True,
            name="idx_feedback_job_id_unique",
        )
        await feedback.create_index(
            [("reviewed_at", DESCENDING)],
            name="idx_feedback_reviewed_time",
        )

        logger.info("MongoDB indexing completed successfully.")
