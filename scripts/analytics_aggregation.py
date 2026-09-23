import asyncio
import json
import logging
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.repositories.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analytics_aggregator")

async def run_aggregation_pipeline() -> None:
    """Aggregates prediction statistics grouped by source platform."""
    await DatabaseManager.connect()
    db = DatabaseManager.get_database()

    pipeline = [
        {
            "$addFields": {
                "string_id": {"$toString": "$_id"}
            }
        },
        {
            "$lookup": {
                "from": "predictions",
                "localField": "string_id",
                "foreignField": "job_id",
                "as": "prediction_data"
            }
        },
        {"$unwind": "$prediction_data"},
        {
            "$group": {
                "_id": "$source_platform",
                "total_submissions": {"$sum": 1},
                "fake_jobs_flagged": {
                    "$sum": {"$cond": [{"$eq": ["$prediction_data.is_fake", True]}, 1, 0]}
                },
                "avg_confidence": {"$avg": "$prediction_data.confidence_score"},
                "review_required_count": {
                    "$sum": {"$cond": [{"$eq": ["$prediction_data.requires_human_review", True]}, 1, 0]}
                }
            }
        },
        {
            "$project": {
                "platform": "$_id",
                "total_submissions": 1,
                "fake_jobs_flagged": 1,
                "fraud_rate_pct": {
                    "$round": [{"$multiply": [{"$divide": ["$fake_jobs_flagged", "$total_submissions"]}, 100]}, 2]
                },
                "avg_confidence": {"$round": ["$avg_confidence", 4]},
                "review_required_count": 1,
                "_id": 0
            }
        },
        {"$sort": {"fraud_rate_pct": -1}}
    ]

    cursor = db["raw_jobs"].aggregate(pipeline)
    results = await cursor.to_list(length=100)

    logger.info("=== Platform Fraud Risk Summary ===")
    print(json.dumps(results, indent=2))

    await DatabaseManager.disconnect()

if __name__ == "__main__":
    asyncio.run(run_aggregation_pipeline())
