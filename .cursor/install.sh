#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT='/agent/repos/Megatron-LM'
if [ ! -d "$REPO_ROOT" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.7.2/install.sh | sh
  sudo ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv
fi

export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
uv venv "${UV_PROJECT_ENVIRONMENT}" --allow-existing

# pyproject.toml override-dependencies omit torch during uv sync; install it explicitly first.
(cd /tmp && uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
  'torch' 'numpy<2.0.0' 'setuptools<80.0.0,>=77.0.0' 'packaging>=24.2')

uv sync --locked --extra lts --extra mlm \
  --only-group linting --only-group test --only-group build --only-group ci

(cd /tmp && uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" torch)
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" -e .

if [ ! -d /opt/data/datasets ] || [ ! -d /opt/data/tokenizers ]; then
  sudo mkdir -p /opt/data
  sudo chown "$(id -u):$(id -g)" /opt/data
  "${UV_PROJECT_ENVIRONMENT}/bin/python" tests/test_utils/python_scripts/download_unit_tests_dataset.py
  cp -an "${REPO_ROOT}/assets/"* /opt/data/
fi