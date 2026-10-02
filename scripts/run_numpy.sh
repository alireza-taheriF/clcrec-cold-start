#!/usr/bin/env bash
# Train and evaluate the from-scratch NumPy encoder, saving API artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py --backend numpy --save-artifacts "$@"
