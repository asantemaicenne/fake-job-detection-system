import argparse
import asyncio
import logging
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.settings import settings
from src.data.repositories.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gdpr_cleanup")

async def execute_gdpr_hard_delete(identifier: str, by_email: bool = True) -> None:
    """Cascade deletes records matching an email or poster_id across all collections."""
    await DatabaseManager.connect()
    db = DatabaseManager.get_database()

    # 1. Identify targeted raw jobs
    field = "contact_email" if by_email else "poster_id"
    query = {field: identifier}
    raw_cursor = db["raw_jobs"].find(query)
    target_jobs = await raw_cursor.to_list(length=10000)

    if not target_jobs:
        logger.warning(f"No records found matching {field}='{identifier}'. No action taken.")
        await DatabaseManager.disconnect()
        return

    job_ids = [str(job["_id"]) for job in target_jobs]
    logger.info(f"Identified {len(job_ids)} job postings associated with {field}='{identifier}'.")

    # 2. Cascade delete
    res_jobs = await db["raw_jobs"].delete_many(query)
    res_features = await db["job_features"].delete_many({"job_id": {"$in": job_ids}})
    res_predictions = await db["predictions"].delete_many({"job_id": {"$in": job_ids}})
    res_feedback = await db["feedback"].delete_many({"job_id": {"$in": job_ids}})

    logger.info(
        f"GDPR Deletion Summary for '{identifier}':\n"
        f" - raw_jobs deleted: {res_jobs.deleted_count}\n"
        f" - job_features deleted: {res_features.deleted_count}\n"
        f" - predictions deleted: {res_predictions.deleted_count}\n"
        f" - feedback records deleted: {res_feedback.deleted_count}"
    )

    await DatabaseManager.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDPR Cascading Hard-Delete Utility")
    parser.add_argument("--id", required=True, help="Email address or poster ID to erase")
    parser.add_argument("--by-poster-id", action="store_true", help="Delete using poster_id instead of email")
    args = parser.parse_args()

    asyncio.run(execute_gdpr_hard_delete(args.id, by_email=not args.by_poster_id))
