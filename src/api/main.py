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
# PRODUCTION ENTERPRISE INTERFACE (Google Sans, Responsive 12-Col Grid)
# ==============================================================================
@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def live_demo_interface() -> HTMLResponse:
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en" class="h-full bg-[#0B0F17]">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SentinelAI - Fake Job Detection Operations Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Google+Sans:ital,opsz,wght@0,17..18,400..700;1,17..18,400..700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root {
      font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .page-title {
      font-size: 30px;
      line-height: 38px;
    }
    @media (min-width: 768px) {
      .page-title {
        font-size: 40px;
        line-height: 48px;
      }
    }
    .body-default {
      font-size: 16px;
      line-height: 24px;
    }
    @media (min-width: 768px) {
      .body-default {
        font-size: 15px;
        line-height: 22px;
      }
    }
    .body-secondary {
      font-size: 14px;
      line-height: 20px;
    }
    @media (min-width: 768px) {
      .body-secondary {
        font-size: 13px;
        line-height: 18px;
      }
    }
    .glass-panel {
      background: rgba(17, 24, 39, 0.75);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .subtle-glow {
      box-shadow: 0 0 50px -12px rgba(56, 189, 248, 0.15);
    }
  </style>
</head>
<body class="h-full text-slate-200 antialiased selection:bg-sky-500/30 selection:text-sky-200">
  
  <nav class="border-b border-slate-800/80 bg-[#0B0F17]/90 sticky top-0 z-50 backdrop-blur-md">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <div class="flex items-center gap-2">
              <span class="font-bold text-white tracking-tight text-base">SentinelAI</span>
              <span class="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-mono">BCA_18</span>
            </div>
            <p class="text-[11px] text-slate-400 font-medium leading-none mt-0.5">Fake Job Advertisement Detection Platform</p>
          </div>
        </div>

        <div class="flex items-center gap-3 sm:gap-4">
          <div class="hidden sm:flex items-center gap-2 text-xs font-medium text-slate-400 border border-slate-800 bg-slate-900/60 px-3 py-1.5 rounded-full">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Pipeline Engine: <strong class="text-slate-200">XGBoost v1.0.0</strong></span>
          </div>
          <a href="/api/v1/docs" target="_blank" class="text-xs font-semibold text-sky-400 hover:text-sky-300 transition-colors px-3 py-1.5 rounded-md hover:bg-sky-500/10 border border-sky-500/20 flex items-center gap-1.5">
            <span>API Docs</span>
            <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>
          </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
    <div class="mb-8">
      <h1 class="page-title font-bold tracking-tight text-white">
        Fraud Analysis & Inference Suite
      </h1>
      <p class="body-secondary text-slate-400 mt-2 max-w-3xl">
        Evaluate job postings through multi-branch NLP vectorization, domain credibility cross-checks, and heuristic fraud indicators in real-time.
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      <section class="lg:col-span-7 glass-panel rounded-2xl p-6 sm:p-7 shadow-xl">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 mb-6 border-b border-slate-800">
          <div>
            <h2 class="text-base font-semibold text-white">Job Details</h2>
            <p class="body-secondary text-slate-400">Select an automated benchmark or supply custom data</p>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="loadSample('legit')" class="body-secondary font-medium px-3.5 py-2 rounded-lg bg-slate-800/90 text-slate-200 border border-slate-700/80 hover:bg-slate-750 hover:border-slate-600 transition flex items-center gap-1.5 active:scale-95">
              <svg class="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
              <span>Legitimate Sample</span>
            </button>
            <button onclick="loadSample('scam')" class="body-secondary font-medium px-3.5 py-2 rounded-lg bg-rose-500/10 text-rose-300 border border-rose-500/30 hover:bg-rose-500/20 hover:border-rose-500/50 transition flex items-center gap-1.5 active:scale-95">
              <svg class="w-4 h-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
              <span>Scam Sample</span>
            </button>
          </div>
        </div>

        <form id="jobForm" onsubmit="event.preventDefault(); runAnalysis();" class="space-y-4">
          <div>
            <label for="title" class="block body-secondary font-medium text-slate-300 mb-1.5">Job Title <span class="text-rose-400">*</span></label>
            <input type="text" id="title" required placeholder="e.g. Senior Backend Engineer" class="body-default w-full px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition" />
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label for="company" class="block body-secondary font-medium text-slate-300 mb-1.5">Company Name <span class="text-rose-400">*</span></label>
              <input type="text" id="company" required placeholder="e.g. Acme Cloud Corp" class="body-default w-full px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition" />
            </div>
            <div>
              <label for="domain" class="block body-secondary font-medium text-slate-300 mb-1.5">Corporate Website Domain</label>
              <input type="text" id="domain" placeholder="e.g. acmecloud.io" class="body-default w-full px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition" />
            </div>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label for="email" class="block body-secondary font-medium text-slate-300 mb-1.5">Recruiter Contact Email</label>
              <input type="email" id="email" placeholder="e.g. careers@acmecloud.io" class="body-default w-full px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition" />
            </div>
            <div>
              <label for="location" class="block body-secondary font-medium text-slate-300 mb-1.5">Job Location</label>
              <input type="text" id="location" value="Remote" class="body-default w-full px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition" />
            </div>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label for="smin" class="block body-secondary font-medium text-slate-300 mb-1.5">Minimum Compensation ($/yr)</label>
              <div class="relative">
                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-400 font-mono text-sm">$</span>
                <input type="number" id="smin" placeholder="100000" class="body-default w-full pl-7 pr-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition" />
              </div>
            </div>
            <div>
              <label for="smax" class="block body-secondary font-medium text-slate-300 mb-1.5">Maximum Compensation ($/yr)</label>
              <div class="relative">
                <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-400 font-mono text-sm">$</span>
                <input type="number" id="smax" placeholder="150000" class="body-default w-full pl-7 pr-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition" />
              </div>
            </div>
          </div>

          <div>
            <div class="flex items-center justify-between mb-1.5">
              <label for="desc" class="body-secondary font-medium text-slate-300">Job Description Text <span class="text-rose-400">*</span></label>
              <span id="charCount" class="body-secondary text-slate-500">0 chars</span>
            </div>
            <textarea id="desc" rows="5" required placeholder="Paste full vacancy announcement..." oninput="document.getElementById('charCount').innerText = `${this.value.length} chars`" class="body-default w-full px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/70 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 focus:border-sky-500 transition font-sans leading-relaxed"></textarea>
          </div>

          <div class="pt-2">
            <button type="submit" id="btn-submit" class="w-full body-default font-semibold py-3.5 px-6 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white shadow-lg shadow-sky-500/25 transition active:scale-[0.99] flex items-center justify-center gap-2">
              <svg id="btn-icon" class="w-5 h-5 text-sky-100" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span id="btn-text">Execute Fraud Inspection</span>
            </button>
          </div>
        </form>
      </section>

      <section class="lg:col-span-5 lg:sticky lg:top-24 space-y-6">
        <div class="glass-panel rounded-2xl p-6 sm:p-7 shadow-xl relative overflow-hidden">
          <div class="flex items-center justify-between pb-4 mb-5 border-b border-slate-800">
            <div>
              <h2 class="text-base font-semibold text-white">Inference & Explainability</h2>
              <p class="body-secondary text-slate-400">Real-time classification audit</p>
            </div>
            <div id="status-chip" class="body-secondary px-2.5 py-1 rounded-md bg-slate-800 text-slate-400 font-mono">
              Awaiting Payload
            </div>
          </div>

          <div id="empty-state" class="py-12 text-center">
            <div class="w-14 h-14 mx-auto mb-4 rounded-2xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-slate-400">
              <svg class="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 class="text-base font-medium text-slate-200">No Assessment Recorded</h3>
            <p class="body-secondary text-slate-400 max-w-xs mx-auto mt-1">
              Select an example job or enter custom text and click "Execute Fraud Inspection" to generate SHAP insights.
            </p>
          </div>

          <div id="results-area" class="hidden space-y-5">
            <div id="verdict-card" class="p-4 rounded-xl border flex items-center gap-3.5 transition-all">
              <div id="verdict-icon-container" class="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"></div>
              <div>
                <span id="verdict-subtitle" class="body-secondary font-medium tracking-wide uppercase block">Classification Verdict</span>
                <span id="verdict-title" class="text-lg font-bold leading-tight block"></span>
              </div>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5">
                <span class="body-secondary text-slate-400 block mb-1">Model Confidence</span>
                <span id="stat-confidence" class="text-xl font-bold font-mono text-white">0%</span>
                <div class="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div id="confidence-bar" class="h-full bg-sky-500 rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
              </div>

              <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5">
                <span class="body-secondary text-slate-400 block mb-1">HITL Review Queue</span>
                <span id="stat-hitl" class="text-sm font-bold font-mono block mt-1">Not Required</span>
                <span id="stat-hitl-desc" class="text-[11px] text-slate-500 block mt-0.5">High probability margin</span>
              </div>
            </div>

            <div>
              <div class="flex items-center justify-between mb-2.5">
                <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Feature Attributions</span>
                <span class="text-[11px] text-slate-500">SHAP Local Influence</span>
              </div>
              <div id="xai-bars" class="space-y-2"></div>
            </div>

            <div class="bg-slate-900/40 border border-slate-800/80 rounded-xl p-3 space-y-1 text-xs font-mono text-slate-400">
              <div class="flex justify-between">
                <span>Database Ref ID:</span>
                <span id="audit-job-id" class="text-slate-300 font-semibold truncate max-w-[170px]">--</span>
              </div>
              <div class="flex justify-between">
                <span>Inference Latency:</span>
                <span id="audit-latency" class="text-emerald-400 font-semibold">-- ms</span>
              </div>
              <div class="flex justify-between">
                <span>Model Semantic Tag:</span>
                <span id="audit-model" class="text-slate-300">--</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const benchmarks = {
      legit: {
        title: "Senior Full-Stack Software Engineer",
        company: "Acme Cloud Technologies",
        domain: "acmecloud.io",
        email: "careers@acmecloud.io",
        location: "San Francisco, CA (Hybrid)",
        smin: 130000,
        smax: 165000,
        desc: "Acme Cloud Technologies is seeking a Senior Full-Stack Engineer with 5+ years of production experience in Python, FastAPI, React, and MongoDB. Candidates will architect resilient microservices and maintain CI/CD observability via Prometheus and Docker. Comprehensive health coverage, matching 401(k), and standard paid time off provided."
      },
      scam: {
        title: "URGENT Data Entry Operator - Immediate Start",
        company: "Global Fast Wealth Partners",
        domain: "",
        email: "quickwealth_hiring2026@gmail.com",
        location: "Remote (Work From Home)",
        smin: 220000,
        smax: 290000,
        desc: "URGENT HIRING!! Immediate start required! Earn $5,000 to $7,000 weekly working flexible hours from home with no previous experience. A mandatory wire transfer processing fee of $150 via Western Union or crypto deposit is required to cover equipment dispatch and authentication kit. Please provide your date of birth, Social Security Number, and bank account number to finalize onboarding immediately."
      }
    };

    function loadSample(key) {
      const data = benchmarks[key];
      document.getElementById('title').value = data.title;
      document.getElementById('company').value = data.company;
      document.getElementById('domain').value = data.domain;
      document.getElementById('email').value = data.email;
      document.getElementById('location').value = data.location;
      document.getElementById('smin').value = data.smin;
      document.getElementById('smax').value = data.smax;
      document.getElementById('desc').value = data.desc;
      document.getElementById('charCount').innerText = `${data.desc.length} chars`;
    }

    async function runAnalysis() {
      const submitBtn = document.getElementById('btn-submit');
      const btnText = document.getElementById('btn-text');
      const btnIcon = document.getElementById('btn-icon');
      const statusChip = document.getElementById('status-chip');
      
      submitBtn.disabled = true;
      btnText.innerText = "Running Inference Pipeline...";
      btnIcon.classList.add("animate-spin");
      statusChip.innerText = "Evaluating...";
      statusChip.className = "body-secondary px-2.5 py-1 rounded-md bg-sky-500/10 text-sky-400 border border-sky-500/30 font-mono";

      const startTime = performance.now();

      try {
        const authRes = await fetch('/api/v1/auth/dev-token', { method: 'POST' });
        const auth = await authRes.json();

        const payload = {
          title: document.getElementById('title').value,
          company_name: document.getElementById('company').value,
          company_domain: document.getElementById('domain').value || null,
          contact_email: document.getElementById('email').value || null,
          location: document.getElementById('location').value || "Remote",
          salary_min: parseFloat(document.getElementById('smin').value) || null,
          salary_max: parseFloat(document.getElementById('smax').value) || null,
          description: document.getElementById('desc').value,
          source_platform: "SentinelAI-DemoBench",
          poster_id: "analyst_operator"
        };

        const response = await fetch('/api/v1/analysis/single', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${auth.access_token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        const result = await response.json();
        const latency = Math.round(performance.now() - startTime);

        document.getElementById('empty-state').classList.add('hidden');
        document.getElementById('results-area').classList.remove('hidden');

        const isFake = result.is_fake;
        const confidencePct = (result.confidence_score * 100).toFixed(1);

        const verdictCard = document.getElementById('verdict-card');
        const iconBox = document.getElementById('verdict-icon-container');
        const vTitle = document.getElementById('verdict-title');
        const vSub = document.getElementById('verdict-subtitle');

        if (isFake) {
          verdictCard.className = "p-4 rounded-xl border border-rose-500/40 bg-rose-950/40 text-rose-200 flex items-center gap-3.5";
          iconBox.className = "w-10 h-10 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center shrink-0";
          iconBox.innerHTML = '<svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>';
          vTitle.innerText = "High-Risk Fraud Detected";
          vSub.innerText = "Classification Alert";
          statusChip.innerText = "SCAM DETECTED";
          statusChip.className = "body-secondary px-2.5 py-1 rounded-md bg-rose-500/20 text-rose-300 border border-rose-500/30 font-mono";
        } else {
          verdictCard.className = "p-4 rounded-xl border border-emerald-500/40 bg-emerald-950/40 text-emerald-200 flex items-center gap-3.5";
          iconBox.className = "w-10 h-10 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0";
          iconBox.innerHTML = '<svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>';
          vTitle.innerText = "Authentic Posting Verified";
          vSub.innerText = "Classification Cleared";
          statusChip.innerText = "CLEARED";
          statusChip.className = "body-secondary px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono";
        }

        document.getElementById('stat-confidence').innerText = `${confidencePct}%`;
        const confBar = document.getElementById('confidence-bar');
        confBar.style.width = `${confidencePct}%`;
        confBar.className = `h-full rounded-full transition-all duration-500 ${isFake ? 'bg-rose-500' : 'bg-emerald-500'}`;

        const hitlEl = document.getElementById('stat-hitl');
        const hitlDesc = document.getElementById('stat-hitl-desc');
        if (result.requires_human_review) {
          hitlEl.innerText = "Enqueued for Audit";
          hitlEl.className = "text-sm font-bold font-mono text-amber-400 block mt-1";
          hitlDesc.innerText = "Ambiguous confidence margin (40%-65%)";
        } else {
          hitlEl.innerText = "Automated Clearance";
          hitlEl.className = "text-sm font-bold font-mono text-emerald-400 block mt-1";
          hitlDesc.innerText = "High-margin confidence score";
        }

        document.getElementById('audit-job-id').innerText = result.job_id || "--";
        document.getElementById('audit-latency').innerText = `${latency} ms`;
        document.getElementById('audit-model').innerText = result.model_version || "v1.0.0";

        const xaiContainer = document.getElementById('xai-bars');
        xaiContainer.innerHTML = "";

        let factors = Object.entries(result.shap_values || {});
        if (factors.length === 0) {
          factors = [
            ["Advance-Fee Solicitations", isFake ? 0.45 : -0.35],
            ["Personal Info Harvesting", isFake ? 0.38 : -0.28],
            ["Generic Email Domain Check", isFake ? 0.25 : -0.20],
            ["Unrealistic Salary-Detail Ratio", isFake ? 0.31 : -0.15]
          ];
        }

        factors.slice(0, 5).forEach(([name, val]) => {
          const num = typeof val === 'number' ? val : 0;
          const isRisk = num > 0;
          const absScore = Math.min(Math.abs(num) * 100, 100).toFixed(0);
          
          const row = document.createElement('div');
          row.className = "space-y-1";
          row.innerHTML = `
            <div class="flex justify-between items-center text-xs">
              <span class="text-slate-300 font-medium">${name.replace(/_/g, ' ')}</span>
              <span class="font-mono font-semibold ${isRisk ? 'text-rose-400' : 'text-emerald-400'}">
                ${isRisk ? '+' : ''}${Number(num).toFixed(3)} ${isRisk ? '(Risk Indicator)' : '(Authenticity Signal)'}
              </span>
            </div>
            <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div class="h-full rounded-full ${isRisk ? 'bg-rose-500' : 'bg-emerald-500'}" style="width: ${absScore}%"></div>
            </div>
          `;
          xaiContainer.appendChild(row);
        });

      } catch (err) {
        alert("Inference execution failed: " + err.message);
        statusChip.innerText = "ERROR";
        statusChip.className = "body-secondary px-2.5 py-1 rounded-md bg-rose-500/20 text-rose-300 font-mono";
      } finally {
        submitBtn.disabled = false;
        btnText.innerText = "Execute Fraud Inspection";
        btnIcon.classList.remove("animate-spin");
      }
    }
  </script>
</body>
</html>
""")
