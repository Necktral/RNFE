#!/usr/bin/env bash
set -euo pipefail

# Ejecuta pytest usando el intérprete del entorno virtual del proyecto
PYTHON_BIN="/home/necktral/.pyenv/versions/rnfe-lab/bin/python"
export PYTHONPATH="src:${PYTHONPATH:-}"

if [ "$#" -eq 0 ]; then
  exec "$PYTHON_BIN" -m pytest -q
else
  exec "$PYTHON_BIN" -m pytest -q "$@"
fi
