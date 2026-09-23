import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    from configs.settings import settings
    from src.features.extractors import (
        ExplicitIndicatorsExtractor,
        LinguisticPatternsExtractor,
        StructuralSignalsExtractor,
    )
    from src.models.training.trainer import ModelTrainer
except ModuleNotFoundError:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from configs.settings import settings
    from src.features.extractors import (
        ExplicitIndicatorsExtractor,
        LinguisticPatternsExtractor,
        StructuralSignalsExtractor,
    )
    from src.models.training.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")


def generate_synthetic_benchmark_dataset(
    sample_size: int = 80,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic training dataset spanning 60 days of
    chronologically ordered job advertisements to enable temporal
    cross-validation out of the box.
    """
    logger.info(
        f"Generating synthetic benchmark dataset ({sample_size} records)..."
    )
    records = []
    base_time = datetime.now(timezone.utc) - timedelta(days=60)

    # Templates for legitimate listings
    legit_templates = [
        (
            "Senior Backend Engineer",
            "Acme Cloud Systems",
            "acmecloud.io",
            (
                "We are seeking an experienced Backend Engineer proficient "
                "in Python, FastAPI, and distributed systems. You will "
                "architect high-throughput APIs and work with MongoDB."
            ),
            "recruiting@acmecloud.io",
            "San Francisco, CA",
            120000.0,
            160000.0,
        ),
        (
            "Frontend UI/UX Developer",
            "Fintech Dynamics",
            "fintechdynamics.com",
            (
                "Join our design-forward engineering department building "
                "responsive React and Next.js applications. Strong "
                "proficiency in TypeScript, CSS architecture, and Figma "
                "required."
            ),
            "talent@fintechdynamics.com",
            "Remote",
            95000.0,
            130000.0,
        ),
        (
            "Data Platform Specialist",
            "Global Analytics Corp",
            "globalanalytics.net",
            (
                "Looking for a Data Engineer with expertise in PySpark, SQL "
                "pipelines, and Kafka. Standard competitive benefits, "
                "retirement matching, and healthcare package included."
            ),
            "careers@globalanalytics.net",
            "New York, NY",
            130000.0,
            175000.0,
        ),
        (
            "DevOps & Security Architect",
            "Nexus Infrastructure",
            "nexusinfra.tech",
            (
                "Managing Kubernetes clusters, Prometheus observability "
                "metrics, and GitHub Actions CI/CD pipelines across multiple "
                "cloud environments."
            ),
            "jobs@nexusinfra.tech",
            "Remote",
            140000.0,
            190000.0,
        ),
    ]

    # Templates for fraudulent/scam listings
    scam_templates = [
        (
            "URGENT: Work from Home Data Entry Operator",
            "Immediate Wealth Enterprises",
            None,
            (
                "URGENT HIRING!! Immediate start required! Make $5000 weekly "
                "with no experience needed. Send wire transfer processing "
                "fee of $150 via Western Union or crypto deposit to secure "
                "starter kit. Submit your Social Security Number and "
                "banking details immediately."
            ),
            "easywealth2026@gmail.com",
            "Remote",
            220000.0,
            300000.0,
        ),
        (
            "Executive VIP Personal Shopper Assistant",
            "Global Logistics Hub",
            None,
            (
                "Earn instant cash today!! We mail cashier's checks directly "
                "to your address. Deposit into your bank account, keep 20%, "
                "and send remainder via Zelle or gift cards. Act fast! "
                "Limited positions open. Guaranteed income."
            ),
            "recruiter_fastjobs@yahoo.com",
            "Remote",
            180000.0,
            240000.0,
        ),
        (
            "Crypto Arbitrage Administrative Clerk",
            "Decentralized Operations Inc",
            None,
            (
                "Urgent vacancy! Convert Bitcoin and USDT transactions from "
                "your personal device. Deposit $200 registration fee before "
                "interview. Send date of birth and passport details."
            ),
            "hiring.desk.desk@hotmail.com",
            "Remote",
            260000.0,
            320000.0,
        ),
        (
            "Remote Package Dispatch Manager",
            "Apex Parcel Forwarding",
            None,
            (
                "Receive parcels at your home and reship immediately. No "
                "background checks required! Apply immediately before "
                "positions fill up. Cash advances paid via MoneyGram."
            ),
            "apex_parcels_hr@outlook.com",
            "Remote",
            150000.0,
            210000.0,
        ),
    ]

    for i in range(sample_size):
        # Alternate between legitimate and fake to maintain a balanced set.
        is_fake = bool(i % 2 == 1)
        template_pool = scam_templates if is_fake else legit_templates
        t = template_pool[i % len(template_pool)]

        # Add chronological jitter
        timestamp = base_time + timedelta(hours=i * (60 * 24 / sample_size))

        record = {
            "title": t[0],
            "company_name": t[1],
            "company_domain": t[2],
            "description": t[3],
            "contact_email": t[4],
            "location": t[5],
            "salary_min": t[6],
            "salary_max": t[7],
            "posted_at": timestamp,
            "source_platform": "Telegram" if is_fake else "LinkedIn",
            "poster_id": (
                f"scammer_{i % 5}" if is_fake else f"recruiter_{i % 15}"
            ),
            "is_fake": int(is_fake),
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def extract_features_into_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the heuristic and NLP extractors across all records to assemble the
    tabular numerical features expected by the scikit-learn ColumnTransformer.
    """
    logger.info(
        "Extracting heuristic, linguistic, and structural feature columns..."
    )

    explicit_records = []
    linguistic_records = []
    structural_records = []

    for _, row in df.iterrows():
        text = str(row.get("description", ""))
        s_min = row.get("salary_min")
        s_max = row.get("salary_max")
        email = row.get("contact_email")
        domain = row.get("company_domain")

        # 1. Explicit features
        exp_feats = ExplicitIndicatorsExtractor.extract(text, s_min, s_max)
        explicit_records.append(exp_feats)

        # 2. Linguistic features
        ling_feats = LinguisticPatternsExtractor.extract(text)
        linguistic_records.append(ling_feats)

        # 3. Structural features
        email_str = str(email).lower() if pd.notnull(email) else ""
        domain_str = str(domain).strip() if pd.notnull(domain) else ""
        is_generic = any(
            email_str.endswith(f"@{gen}")
            for gen in StructuralSignalsExtractor.GENERIC_DOMAINS
        )
        missing_url = len(domain_str) == 0

        structural_records.append(
            {
                "is_generic_email": is_generic,
                "missing_company_url": missing_url,
            }
        )

    # 4. Network and duplicate features
    exp_df = pd.DataFrame(explicit_records)
    ling_df = pd.DataFrame(linguistic_records)
    struct_df = pd.DataFrame(structural_records)

    # Poster frequency heuristic for reputation scoring
    poster_counts = df["poster_id"].value_counts().to_dict()
    poster_reputation = [
        max(0.1, 1.0 - (max(0, poster_counts.get(pid, 1) - 5) * 0.05))
        for pid in df["poster_id"]
    ]

    # Duplicate title/company checks
    dup_counts = (
        df.duplicated(subset=["title", "company_name"], keep=False)
        .astype(int)
    )

    enriched_df = pd.concat(
        [df.reset_index(drop=True), exp_df, ling_df, struct_df], axis=1
    )
    enriched_df["poster_reputation_score"] = poster_reputation
    enriched_df["duplicate_count"] = dup_counts.values

    return enriched_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train and optimize the "
            "Fake Job Detection XGBoost Model"
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw CSV dataset (e.g. Kaggle fake_job_postings.csv)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help=(
            "Number of Optuna hyperparameter optimization trials "
            "(default: 15)"
        ),
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=3,
        help=(
            "Number of TimeSeriesSplit folds for temporal "
            "cross-validation (default: 3)"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Target output path for the serialized .pkl model artifact",
    )
    args = parser.parse_args()

    # 1. Load or synthesize dataset
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"Loading raw dataset from {args.data_path}...")
        raw_df = pd.read_csv(args.data_path)
        # Harmonize Kaggle column names if present
        if "fraudulent" in raw_df.columns and "is_fake" not in raw_df.columns:
            raw_df.rename(columns={"fraudulent": "is_fake"}, inplace=True)
        if "posted_at" not in raw_df.columns:
            # Generate pseudo-temporal dates for time-series splits
            base = datetime.now(timezone.utc) - timedelta(days=len(raw_df))
            raw_df["posted_at"] = [
                base + timedelta(days=i) for i in range(len(raw_df))
            ]
        if "poster_id" not in raw_df.columns:
            raw_df["poster_id"] = "unknown_poster"
    else:
        logger.info(
            "No external dataset provided. Using built-in synthetic "
            "benchmark data."
        )
        raw_df = generate_synthetic_benchmark_dataset(sample_size=80)

    # 2. Extract feature columns required by the preprocessor
    full_df = extract_features_into_dataframe(raw_df)

    # 3. Initialize ModelTrainer with TimeSeries cross-validation
    logger.info("Initializing ModelTrainer with temporal cross-validation...")
    trainer = ModelTrainer(
        data=full_df,
        target_column="is_fake",
        timestamp_column="posted_at",
        text_column="description",
    )

    # 4. Run Optuna tuning sweep
    pipeline, best_params, best_f1 = trainer.train_and_tune(
        n_trials=args.trials,
        n_splits=args.splits,
    )
    logger.info(
        f"Optimization complete. Best Mean Temporal F1 Score: {best_f1:.4f}"
    )
    logger.info(f"Best Hyperparameters: {best_params}")

    # 5. Persist artifact to disk
    target_artifact = args.output_path or os.path.join(
        settings.MODEL_ARTIFACTS_DIR,
        f"{settings.MODEL_VERSION}.pkl",
    )
    saved_file = trainer.save_artifact(pipeline, output_path=target_artifact)

    file_size_kb = os.path.getsize(saved_file) / 1024
    logger.info(
        "SUCCESS: Model artifact '%s' saved to: %s (%0.2f KB)",
        settings.MODEL_VERSION,
        saved_file,
        file_size_kb,
    )


if __name__ == "__main__":
    main()
