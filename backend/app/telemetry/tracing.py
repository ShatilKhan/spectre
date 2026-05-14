"""OpenTelemetry instrumentation setup.

Initializes tracing for FastAPI, ChromaDB, and custom spans.
Traces are exported to the OTel collector for visualization in Jaeger.
"""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(app, service_name: str = "spectre-backend"):
    """Configure OpenTelemetry tracing for the FastAPI application.

    Args:
        app: The FastAPI application instance.
        service_name: Name for this service in traces.
    """
    endpoint = os.getenv("OTLP_ENDPOINT", "http://otel-collector:4317")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
