"""OpenTelemetry instrumentation setup.

Initializes tracing for FastAPI and custom spans.
Traces are exported to Jaeger via OTLP gRPC.

Usage at module level (after app creation, before routes):
    setup_tracing()                         # sets up tracer provider
    FastAPIInstrumentor.instrument_app(app) # patches routes

Both are no-ops if OTEL_EXPORTER_OTLP_ENDPOINT is not set.
"""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(service_name: str | None = None) -> bool:
    """Configure the global OpenTelemetry tracer provider.

    Uses the standard OTEL_EXPORTER_OTLP_ENDPOINT env var. If not set,
    this is a silent no-op — the app runs without tracing.

    Args:
        service_name: Override for the service name. Falls back to
                      OTEL_SERVICE_NAME env var, then 'spectre-backend'.

    Returns:
        True if tracing was enabled, False if skipped (no endpoint).
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        return False

    name = service_name or os.getenv("OTEL_SERVICE_NAME", "spectre-backend")
    resource = Resource.create({"service.name": name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    print(f"telemetry: tracing enabled -> {endpoint} (service={name})")
    return True
