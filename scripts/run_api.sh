#!/usr/bin/env bash
# Launch the FastAPI recommendation service.
#
# Requires artifacts to exist first:
#   bash scripts/run_numpy.sh       # or run_torch.sh
#
# Override host/port with RECSYS_API_HOST / RECSYS_API_PORT.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m recsys.api.fastapi_app
