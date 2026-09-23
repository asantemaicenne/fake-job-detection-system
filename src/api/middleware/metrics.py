import time
from typing import Callable
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

# 1. Throughput counter labeled by HTTP method, path, and status code
HTTP_REQUESTS_TOTAL = Counter(
    "api_http_requests_total",
    "Total volume of incoming HTTP requests",
    ["method", "endpoint", "status_code"],
)

# 2. Latency histogram configured with millisecond-to-second response time buckets
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "api_http_request_duration_seconds",
    "HTTP request latency breakdown in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# 3. Model inference distribution counter for drift and fake-job ratio monitoring
MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Number of inference decisions by outcome and review flag",
    ["model_version", "is_fake", "requires_human_review"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware that instruments request timing and throughput metrics.
    Excludes the scraping endpoint itself to prevent skewing telemetry.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        endpoint = request.url.path

        # Bypass telemetry collection on internal metrics scrapes
        if endpoint == "/metrics":
            return await call_next(request)

        method = request.method
        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            raise
        finally:
            duration = time.perf_counter() - start_time
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)


def metrics_endpoint() -> Response:
    """Exports raw Prometheus metrics format to the Prometheus collector."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
