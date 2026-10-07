#!/usr/bin/env bash
# Start the server (settings in .env): UI at http://HOST:PORT/, API docs at /docs
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo ".venv not found - run ./install.sh first"; exit 1; }
exec .venv/bin/python app.py "$@"
