#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Query case: fixed=img_query, moving=img_support
echo "=== Step 1a: Registration for query (fixed=query, moving=support) ==="
python step1_registration.py \
  --fixed-image test_data/images/img_query.nii.gz \
  --moving-image test_data/images/img_support.nii.gz \
  --fixed-label test_data/labels/label_query.nii.gz \
  --moving-label test_data/labels/label_support.nii.gz \
  --out-dir test_data/step1_output

# Support case: fixed=img_support, moving=img_query
echo "=== Step 1b: Registration for support (fixed=support, moving=query) ==="
python step1_registration.py \
  --fixed-image test_data/images/img_support.nii.gz \
  --moving-image test_data/images/img_query.nii.gz \
  --fixed-label test_data/labels/label_support.nii.gz \
  --moving-label test_data/labels/label_query.nii.gz \
  --out-dir test_data/step1_output_support

if [[ ! -f "fm_models/nnInteractive_v1.0/fold_0/checkpoint_final.pth" ]]; then
  echo "Error: nnInteractive model not found."
  echo "Download it to fm_models/nnInteractive_v1.0/ — see README.md"
  exit 1
fi

echo "=== Step 2a: FM segmentation on query ==="
python step2_FM_segment.py \
  --reg-dir test_data/step1_output \
  --out-dir test_data/step2_output \
  --eval-dice

echo "=== Step 2b: FM segmentation on support ==="
python step2_FM_segment.py \
  --reg-dir test_data/step1_output_support \
  --out-dir test_data/step2_output_support \
  --eval-dice

echo "=== Step 3: Fusion adapter (train on support, apply to query) ==="
python step3_fusion_adapter.py --device cuda:3 --full-res

echo "=== Pipeline finished ==="
