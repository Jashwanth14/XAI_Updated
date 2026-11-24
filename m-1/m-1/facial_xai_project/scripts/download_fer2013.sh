#!/usr/bin/env bash
set -euo pipefail

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install it with: python -m pip install kaggle" >&2
  exit 1
fi

mkdir -p data/fer2013
kaggle datasets download -d msambare/fer2013 --path data/fer2013 --unzip
echo "Downloaded under data/fer2013. Review the dataset page terms before training or sharing files."
