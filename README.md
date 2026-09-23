# SentinelAI: AI-Based Fake Job Advertisement Detection System

[![Release](https://img.shields.io/badge/Release-v1.0.1-blue.svg)](https://github.com/asantemaicenne/fake-job-detection-system/releases/tag/v1.0.1)
[![CI Pipeline](https://github.com/asantemaicenne/fake-job-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/asantemaicenne/fake-job-detection-system/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0-47A248.svg)](https://www.mongodb.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/asantemaicenne/fake-job-detection-system/blob/main/LICENSE)

An enterprise-grade, end-to-end Machine Learning, NLP, and Explainable AI (XAI) platform designed to detect fraudulent job advertisements, combat recruitment scams, and protect candidates. Engineered under **Project BCA_18** (Supervisor: **Dr. Priya Dhir**) aligned with **UN SDG 8 (Decent Work and Economic Growth)**.

---

## System Architecture

```mermaid
flowchart TD
    Client["Client (Web Demo / REST API / Bulk Upload)"]
    
    subgraph Gateway_Layer ["FastAPI Application Gateway"]
        Gateway["REST API Router & WebSocket Manager"]
        Sanity["Pre-Inference Validation Layer<br/>(RFC FQDN + Lexical Shannon Entropy)"]
        Security["JWT Authentication & RBAC Guards"]
        Telemetry["Prometheus Metrics Middleware"]
        Gateway --> Sanity
        Sanity --> Security
        Security --> Telemetry
    end
    
    subgraph ML_Pipeline ["Machine Learning & Explainability Pipeline"]
        Vector["Dual-Branch Feature Pipeline<br/>(TF-IDF N-Grams + Numerical Scaler)"]
        Engine["XGBoost Ensemble Classifier (v1.0.0)<br/>Optuna Tuned + TimeSeriesSplit CV"]
        SHAP["SHAP TreeExplainer Attribution Engine"]
        HITL{"HITL Uncertainty Queue<br/>(0.40 <= p <= 0.65)"}
        Vector --> Engine
        Engine --> SHAP
        Engine --> HITL
    end
    
    subgraph Data_Layer ["MongoDB Clustered Datastore"]
        Raw[("raw_jobs<br/>Text Search & Platform Compound Index")]
        Feat[("job_features<br/>Unique Feature Vectors")]
        Pred[("predictions<br/>365-Day TTL Expiration Index")]
        Audit[("feedback<br/>HITL Gold-Standard Verified Store")]
    end
    
    subgraph Observability ["Observability & Alerting Stack"]
        Prom["Prometheus TSDB Engine"]
        Alert["Alertmanager (Webhook & Notification Relay)"]
        Graf["Grafana Operational Dashboards"]
        Prom --> Alert
        Prom --> Graf
    end

    Client -->|"POST /analysis/single"| Gateway
    Telemetry -->|"Persist Ingestion"| Raw
    Telemetry -->|"Execute Extraction"| Vector
    Vector -->|"Store Feature Record"| Feat
    Engine -->|"Store Prediction Result"| Pred
    HITL -->|"Audit Corrections"| Audit
    Telemetry -.->|"Scrape Telemetry"| Prom
end
```

## Core Capabilities & Hardened Security (v1.0.1)

- Adversarial Noise & Lexical Entropy Guards: Pre-inference verification computes Shannon character entropy and filters zero-vector out-of-vocabulary (OOV) inputs, preventing synthetic gibberish from bypassing gradient-boosted trees.
- RFC 1035 Domain & FQDN Validation: Strict verification ensures corporate domains contain recognized public Top-Level Domains (TLDs).
- Multi-Branch NLP Preprocessing: Merges TF-IDF bi-grams with engineered heuristics (advance-fee solicitations, PII theft keywords, grammatical anomalies, and salary-to-effort anomalies) via scikit-learn ColumnTransformer.
- Leakage-Free Cross-Validation: Employs TimeSeriesSplit across chronological splits during Optuna optimization to prevent lookahead bias.
- Explainable AI (XAI): Generates local signed attribution weights for each prediction via shap.TreeExplainer, fulfilling GDPR Right-to-Explanation requirements.
- Human-in-the-Loop (HITL) Routing: Borderline confidence margins ($0.40 \le p \le 0.65$) are quarantined for human review.
- Full Observability Stack: Exposes real-time throughput, latency, and fraud ratio spikes via Prometheus, Alertmanager, and Grafana.
- Cross-Platform Setup & Requirements

## System Prerequisites Across All Devices

- Python: 3.10 or 3.11
- Docker: Docker Desktop (Windows/macOS) or Docker Engine + Docker Compose v2 (Linux)
- Git: 2.30+
- Hardware Requirements: Minimum 4 GB RAM, 2 CPU cores, 10 GB disk space.

### Option A: Windows Setup (PowerShell)

Clone the repository:

```powershell
git clone https://github.com/your-username/fake-job-detection-system.git
Set-Location -Path "fake-job-detection-system"
```

Initialize Python Virtual Environment:

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Spin up Docker Services (MongoDB, Prometheus, Alertmanager, Grafana):

```powershell
docker-compose up -d mongodb mongodb-exporter prometheus alertmanager grafana
```

Train the ML Artifact:

```powershell
python scripts/train_model.py --trials 15 --splits 3
```

Start the API Server:

```powershell
uvicorn src.api.main:app --reload --port 8000
```

### Option B: macOS Setup (Terminal / Zsh)

Install prerequisites (via Homebrew):

```bash
brew install python@3.10 git
# Install Docker Desktop for Mac if not already installed
```

Clone and navigate:

```bash
git clone https://github.com/your-username/fake-job-detection-system.git
cd fake-job-detection-system
```

Initialize environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start infrastructure containers:

```bash
docker compose up -d mongodb mongodb-exporter prometheus alertmanager grafana
```

Train model and run the server:

```bash
python scripts/train_model.py --trials 15 --splits 3
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option C: Linux / Ubuntu / Debian Setup (Bash)

Install system dependencies:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git curl build-essential
# Ensure Docker and Docker Compose plugin are installed
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
```

Clone and configure:

```bash
git clone https://github.com/your-username/fake-job-detection-system.git
cd fake-job-detection-system
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Launch Docker stack:

```bash
sudo docker compose up -d mongodb mongodb-exporter prometheus alertmanager grafana
```

Train artifact and launch service:

```bash
python scripts/train_model.py --trials 15 --splits 3
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option D: Fully Containerized Deployment (Docker Only)

Run the entire platform (API, Database, Observability) inside containers without local Python configuration:

```bash
docker compose up -d --build
```

## API Reference

| Method | Endpoint | Access Role | Description |
| --- | --- | --- | --- |
| GET | /health | Public | System liveness probe and active model semantic tag |
| GET | /metrics | Public / Scraper | Prometheus metrics export endpoint |
| GET | /demo | Public | Interactive Google Sans operations testbench |
| POST | /api/v1/auth/dev-token | Public | Development JWT authentication issuer |
| POST | /api/v1/analysis/single | admin, analyst, api_user | Real-time single job fraud and XAI assessment |
| POST | /api/v1/analysis/bulk | admin, analyst | Batch job submission (up to 100 postings) |
| POST | /api/v1/analysis/feedback | admin, analyst | Human-in-the-loop audit verification input |
| GET | /api/v1/analysis/results | admin, analyst | Paginated prediction querying with outcome filters |
| WS | /api/v1/analysis/ws/status | Public | WebSocket channel for real-time progress updates |

Interactive OpenAPI documentation is accessible at http://localhost:8000/api/v1/docs.

## Data Schema & Retention Policy

| Collection | Key Indexed Fields | Storage Category | Retention Lifecycle |
| --- | --- | --- | --- |
| raw_jobs | description (text), source_platform, posted_at | Active Store | 180 days hot storage |
| job_features | job_id (unique) | Feature Store | Linked to raw posting |
| predictions | job_id (unique), timestamp | Analytical Store | 365-day automated TTL index |
| feedback | job_id (unique), reviewed_at | Gold-Standard Store | Permanent audit retention |

## Verification & Automated Test Suites

The test suite validates input sanitization, adversarial noise immunity, and prediction accuracy:

```powershell
# Run the full integration and adversarial test suite
pytest tests/ -v --durations=10
```

## Academic Alignment & Research Deliverables

- Project Identification: BCA_18
- Project Guide / Supervisor: Dr. Priya Dhir
- United Nations SDG Goal: Goal 8 - Decent Work and Economic Growth
- Target Publication Outlets: SCI / Scopus Indexed Journals, IEEE Conference Proceedings, and Patent Filing
- License: Distributed under the MIT License. See [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/asantemaicenne/fake-job-detection-system/blob/main/LICENSE) for details.
