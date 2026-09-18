#!/usr/bin/env bash
# Download and unpack MNIST into ./data.
#
# The canonical yann.lecun.com URLs are unreliable these days, so we use the
# mirror the PyTorch project maintains.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)

for f in "${FILES[@]}"; do
  if [ -f "data/$f" ]; then
    echo "have  data/$f"
    continue
  fi
  echo "get   $f"
  curl -fsSL "$BASE/$f.gz" -o "data/$f.gz"
  gunzip -f "data/$f.gz"
done

echo
ls -lh data/*-ubyte
