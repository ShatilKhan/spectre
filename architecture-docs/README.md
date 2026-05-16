# Spectre — Architecture Docs

Diagrams are authored in **[D2](https://d2lang.com)** and pre-rendered to SVG + PNG.

## Contents

| File | What it covers |
|------|----------------|
| [`01-overview.md`](./01-overview.md) | System context, request lifecycle, three wire contracts, known issues |
| [`02-integration.md`](./02-integration.md) | Module/component maps, API contracts, data flow diagrams |
| [`03-frontend-flow.md`](./03-frontend-flow.md) | UI tab structure, SSE stream processing, state management |
| [`04-deployment.md`](./04-deployment.md) | Docker Compose topology, environment matrix, Jaeger telemetry |

## Diagram Source

| File | Preview |
|------|---------|
| [`01-context.d2`](./diagrams/01-context.d2) | ![01-context](./diagrams/out/01-context.png) |
| [`02-components.d2`](./diagrams/02-components.d2) | ![02-components](./diagrams/out/02-components.png) |
| [`03-sequence.d2`](./diagrams/03-sequence.d2) | ![03-sequence](./diagrams/out/03-sequence.png) |
| [`05-deployment.d2`](./diagrams/05-deployment.d2) | ![05-deployment](./diagrams/out/05-deployment.png) |

## Re-rendering

```bash
# Install D2 (single Go binary, no system deps)
curl -fsSL https://d2lang.com/install.sh | sh -s --

# Re-render all diagrams
./architecture-docs/diagrams/render.sh
```

Requires `d2` on PATH. ImageMagick optional (for 800x418 blog-sized PNGs).
