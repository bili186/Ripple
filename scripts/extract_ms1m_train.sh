#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
REC="$ROOT_DIR/data/train/faces_ms1m_112x112/train.rec"
IDX="$ROOT_DIR/data/train/faces_ms1m_112x112/train.idx"
OUT_DIR="$ROOT_DIR/data/train/ms1m_refine/ms1m_refine_112x112"
LST="$ROOT_DIR/data/train/ms1m_refine/ms1m_refine.lst"

mkdir -p "$ROOT_DIR/data/train/ms1m_refine"
python3 "$ROOT_DIR/scripts/extract_ms1m_train.py" \
  --rec "$REC" \
  --idx "$IDX" \
  --output_dir "$OUT_DIR" \
  --lst_path "$LST"
