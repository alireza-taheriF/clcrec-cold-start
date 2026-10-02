#!/usr/bin/env bash
# Train and evaluate the PyTorch encoder, saving API artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py --backend torch --save-artifacts "$@"
