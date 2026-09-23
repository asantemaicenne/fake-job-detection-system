import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from configs.settings import settings
from src.api.endpoints import analysis
from src.api.middleware.auth import Token, create_access_token
from src.api.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from src.data.repositories.database import DatabaseManager
from src.models.ethics.safeguards import ExplainabilityEngine, HumanInTheLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup, model loading, and shutdown for the API."""
    # 1. Connect MongoDB
    logger.info("Connecting to MongoDB datastore...")
    await DatabaseManager.connect()

    # 2. Load serialized XGBoost Pipeline artifact
    artifact_path = os.path.join(
        settings.MODEL_ARTIFACTS_DIR,
        f"{settings.MODEL_VERSION}.pkl",
    )
    if os.path.exists(artifact_path):
        try:
            logger.info("Loading model artifact: %s", artifact_path)
            app.state.model_pipeline = joblib.load(artifact_path)
            logger.info("XGBoost pipeline loaded into memory.")

            # 3. Initialize ExplainabilityEngine with background reference data
            if settings.ENABLE_SHAP_EXPLAINABILITY:
                logger.info("Initializing SHAP Explainability Engine...")
                background_baseline = pd.DataFrame(
                    [
                        {
                            "description": (
                                "Standard software engineer with python and "
                                "cloud background"
                            ),
                            "has_payment_request": False,
                            "has_pii_request": False,
                            "salary_anomaly_score": 0.0,
                            "urgency_score": 0.0,
                            "grammar_anomaly_score": 0.0,
                            "is_generic_email": False,
                            "missing_company_url": False,
                            "poster_reputation_score": 1.0,
                            "duplicate_count": 0,
                        },
                        {
                            "description": (
                                "Urgent hiring wire transfer payment upfront "
                                "needed immediately"
                            ),
                            "has_payment_request": True,
                            "has_pii_request": True,
                            "salary_anomaly_score": 1.0,
                            "urgency_score": 0.8,
                            "grammar_anomaly_score": 0.5,
                            "is_generic_email": True,
                            "missing_company_url": True,
                            "poster_reputation_score": 0.1,
                            "duplicate_count": 5,
                        },
                    ]
                )
                app.state.explainer = ExplainabilityEngine(
                    pipeline=app.state.model_pipeline,
                    background_sample=background_baseline,
                )
                logger.info("SHAP Explainer ready for live inference.")
            else:
                app.state.explainer = None
        except (
            AttributeError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error(
                "Failed to initialize ML models: %s. Falling back to "
                "heuristics.",
                exc,
            )
            app.state.model_pipeline = None
            app.state.explainer = None
    else:
        logger.warning(
            "No artifact found at '%s'. Using heuristic fallback.",
            artifact_path,
        )
        app.state.model_pipeline = None
        app.state.explainer = None

    app.state.hitl = HumanInTheLoop(
        lower_threshold=settings.HITL_UNCERTAINTY_LOWER,
        upper_threshold=settings.HITL_UNCERTAINTY_UPPER,
    )
    logger.info("Application startup complete.")

    yield

    # 4. Graceful shutdown
    logger.info("Disconnecting from MongoDB...")
    await DatabaseManager.disconnect()
    logger.info("Application shutdown completed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.MODEL_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.PROMETHEUS_METRICS_ENABLED:
    app.add_middleware(PrometheusMiddleware)
    app.add_api_route(
        "/metrics",
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
    )

app.include_router(
    analysis.router,
    prefix=f"{settings.API_V1_STR}/analysis",
    tags=["Job Analysis & Predictions"],
)


@app.get("/health", tags=["System"], status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.MODEL_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.post(
    f"{settings.API_V1_STR}/auth/dev-token",
    response_model=Token,
    tags=["Authentication"],
    status_code=status.HTTP_200_OK,
)
async def generate_development_token(
    username: str = "dev_admin",
    role: str = "admin",
) -> Token:
    access_token = create_access_token(subject=username, role=role)
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ============================================================================== 
# VISUAL DEMONSTRATION TESTBENCH FOR SUPERVISOR DEMO
# ==============================================================================
@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def live_demo_interface() -> HTMLResponse:
    return HTMLResponse(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>AI Fake Job Detection System - Live Demo</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen p-6 font-sans">
      <div class="max-w-4xl mx-auto">
        <header class="border-b border-slate-700 pb-4 mb-6 flex justify-between items-center">
          <div>
            <h1 class="text-2xl font-bold text-sky-400">AI-Based Fake Job Detection System</h1>
            <p class="text-xs text-slate-400">Production-ready AI & NLP system to detect fraudulent job advertisements and recruitment scams using FastAPI, XGBoost, Optuna, MongoDB, and Prometheus/Grafana observability.</p>
          </div>
          <span class="bg-emerald-950 text-emerald-400 border border-emerald-800 text-xs px-2.5 py-1 rounded-full font-mono">v1.0.0 Online</span>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="bg-slate-800/80 p-5 rounded-xl border border-slate-700">
            <h2 class="text-base font-semibold mb-3">Job Advertisement Input</h2>
            <div class="flex gap-2 mb-3">
              <button onclick="loadSample('legit')" class="text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded transition">Load Legitimate Job</button>
              <button onclick="loadSample('scam')" class="text-xs bg-rose-950 text-rose-300 border border-rose-800 hover:bg-rose-900 px-3 py-1.5 rounded transition">Load Scam Job</button>
            </div>
            <div class="space-y-3">
              <div>
                <label class="text-xs text-slate-400">Job Title</label>
                <input id="title" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-200" />
              </div>
              <div>
                <label class="text-xs text-slate-400">Company Name & Domain</label>
                <div class="grid grid-cols-2 gap-2">
                  <input id="company" placeholder="Company" class="bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-200" />
                  <input id="domain" placeholder="Domain (optional)" class="bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-200" />
                </div>
              </div>
              <div>
                <label class="text-xs text-slate-400">Contact Email</label>
                <input id="email" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-200" />
              </div>
              <div class="grid grid-cols-2 gap-2">
                <div>
                  <label class="text-xs text-slate-400">Min Salary ($)</label>
                  <input id="smin" type="number" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-200" />
                </div>
                <div>
                  <label class="text-xs text-slate-400">Max Salary ($)</label>
                  <input id="smax" type="number" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-200" />
                </div>
              </div>
              <div>
                <label class="text-xs text-slate-400">Job Description Body</label>
                <textarea id="desc" rows="5" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-200"></textarea>
              </div>
              <button onclick="runAnalysis()" id="btn-submit" class="w-full bg-sky-600 hover:bg-sky-500 font-semibold py-2.5 rounded text-sm transition">Analyze with Machine Learning</button>
            </div>
          </div>

          <div class="bg-slate-800/80 p-5 rounded-xl border border-slate-700 flex flex-col">
            <h2 class="text-base font-semibold mb-3">Model Inference & Explainability</h2>
            <div id="results-placeholder" class="text-slate-500 text-center my-auto py-12 text-sm">
              Submit a job posting or load a sample to inspect real-time AI predictions.
            </div>
            <div id="results-content" class="hidden space-y-4">
              <div id="verdict-banner" class="p-4 rounded-lg text-center font-bold text-lg border"></div>
              <div class="bg-slate-900/60 p-3 rounded border border-slate-700 text-xs space-y-1 font-mono">
                <div><strong>Confidence:</strong> <span id="res-conf"></span></div>
                <div><strong>Requires HITL Review:</strong> <span id="res-hitl"></span></div>
                <div><strong>Database Reference ID:</strong> <span id="res-id" class="text-slate-400"></span></div>
              </div>
              <div>
                <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">SHAP Feature Attribution (Why the model made this call)</h3>
                <div id="shap-container" class="space-y-1.5 text-xs"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <script>
        const samples = {
          legit: {
            title: "Senior Backend Developer",
            company: "Acme Cloud Corp",
            domain: "acmecloud.io",
            email: "careers@acmecloud.io",
            smin: 120000,
            smax: 150000,
            desc: "We are seeking an experienced Backend Engineer skilled in Python, FastAPI, and MongoDB. Candidates should understand distributed systems, clean architecture, and CI/CD pipelines. Standard health, dental, and 401(k) benefits."
          },
          scam: {
            title: "URGENT Data Entry - Immediate Start",
            company: "Quick Wealth Partners",
            domain: "",
            email: "recruiter991@gmail.com",
            smin: 220000,
            smax: 280000,
            desc: "URGENT HIRING!! Immediate start required! Make $5,000 weekly from home. Send a wire transfer processing fee of $150 via Western Union or crypto for your starter kit. Send your Social Security Number and banking details to apply."
          }
        };

        function loadSample(type) {
          const s = samples[type];
          document.getElementById('title').value = s.title;
          document.getElementById('company').value = s.company;
          document.getElementById('domain').value = s.domain;
          document.getElementById('email').value = s.email;
          document.getElementById('smin').value = s.smin;
          document.getElementById('smax').value = s.smax;
          document.getElementById('desc').value = s.desc;
        }

        async function runAnalysis() {
          const btn = document.getElementById('btn-submit');
          btn.innerText = "Analyzing...";
          btn.disabled = true;

          try {
            const authRes = await fetch('/api/v1/auth/dev-token', { method: 'POST' });
            const authData = await authRes.json();

            const payload = {
              title: document.getElementById('title').value,
              company_name: document.getElementById('company').value,
              company_domain: document.getElementById('domain').value || null,
              contact_email: document.getElementById('email').value || null,
              salary_min: parseFloat(document.getElementById('smin').value) || null,
              salary_max: parseFloat(document.getElementById('smax').value) || null,
              description: document.getElementById('desc').value,
              source_platform: "WebDemo",
              poster_id: "demo_supervisor"
            };

            const res = await fetch('/api/v1/analysis/single', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${authData.access_token}`,
                'Content-Type': 'application/json'
              },
              body: JSON.stringify(payload)
            });
            const data = await res.json();

            document.getElementById('results-placeholder').classList.add('hidden');
            document.getElementById('results-content').classList.remove('hidden');

            const banner = document.getElementById('verdict-banner');
            if (data.is_fake) {
              banner.className = "p-4 rounded-lg text-center font-bold text-lg border bg-rose-950/80 border-rose-700 text-rose-300";
              banner.innerText = "⚠️ FRAUDULENT JOB DETECTED (SCAM)";
            } else {
              banner.className = "p-4 rounded-lg text-center font-bold text-lg border bg-emerald-950/80 border-emerald-700 text-emerald-300";
              banner.innerText = "✅ VERIFIED LEGITIMATE JOB";
            }

            document.getElementById('res-conf').innerText = `${(data.confidence_score * 100).toFixed(2)}%`;
            document.getElementById('res-hitl').innerText = data.requires_human_review ? "YES (Needs Review)" : "NO (High Confidence)";
            document.getElementById('res-id').innerText = data.job_id;

            const shapBox = document.getElementById('shap-container');
            shapBox.innerHTML = '';
            for (const [feat, val] of Object.entries(data.shap_values || {})) {
              const isRisk = val > 0;
              const row = document.createElement('div');
              row.className = "flex justify-between items-center py-1 border-b border-slate-700";
              row.innerHTML = `
                <span class="text-slate-300">${feat}</span>
                <span class="font-mono font-bold ${isRisk ? 'text-rose-400' : 'text-emerald-400'}">
                  ${isRisk ? '+' : ''}${val.toFixed(3)} (${isRisk ? 'Risk Factor' : 'Safe Factor'})
                </span>
              `;
              shapBox.appendChild(row);
            }
          } catch (err) {
            alert("Error running inference: " + err.message);
          } finally {
            btn.innerText = "Analyze with Machine Learning";
            btn.disabled = false;
          }
        }
      </script>
    </body>
    </html>
    """
    )
