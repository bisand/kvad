#!/usr/bin/env bash
# Download a text to train on into ./data/input.txt.
#
# The default is the "tiny Shakespeare" file from Andrej Karpathy's char-rnn:
# about 1.1 MB, 40,000 lines of plays. It is what nearly every small language
# model tutorial trains on, so there are plenty of loss curves to compare
# against. Any plain text file works just as well:
#
#     cargo run --release -p nanograd --bin train_text -- --data my.txt
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if [ -f data/input.txt ]; then
  echo "have  data/input.txt"
else
  echo "get   input.txt"
  curl -fsSL "$URL" -o data/input.txt
fi

echo
wc -c data/input.txt
