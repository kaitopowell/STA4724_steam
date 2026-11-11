#!/usr/bin/env bash
# Creates a venv in the repo root and installs requirements.
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
