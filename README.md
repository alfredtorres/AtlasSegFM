<div align="center">

# Atlas is Your Perfect Context

### One-Shot Customization for Generalizable Foundational Medical Image Segmentation

**ECCV 2026 Spotlight**

[![arXiv](https://img.shields.io/badge/arXiv-2512.18176-b31b1b.svg)](https://arxiv.org/abs/2512.18176)
[![Paper](https://img.shields.io/badge/Paper-PDF-green.svg)](https://arxiv.org/pdf/2512.18176)
[![ECCV](https://img.shields.io/badge/ECCV-2026%20Spotlight-orange.svg)](https://eccv.ecva.net/virtual/2026/spotlight/6050)
[![Poster](https://img.shields.io/badge/ECCV-Poster-blue.svg)](https://eccv.ecva.net/virtual/2026/poster/3573)
[![GitHub](https://img.shields.io/badge/GitHub-alfredtorres%2FAtlasSegFM-black.svg)](https://github.com/alfredtorres/AtlasSegFM)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c.svg)](https://pytorch.org/)

Ziyu Zhang<sup>1*</sup>
&nbsp; Yi Yu<sup>2*</sup>
&nbsp; Simeng Zhu<sup>2</sup>
&nbsp; Ahmed Aly<sup>2</sup>
&nbsp; Yunhe Gao<sup>3</sup>
&nbsp; Ning Gu<sup>1†</sup>
&nbsp; Yuan Xue<sup>2†</sup>

<sup>1</sup>Nanjing University &emsp;
<sup>2</sup>The Ohio State University &emsp;
<sup>3</sup>Stanford University

<sup>*</sup>Equal contribution &emsp; <sup>†</sup>Corresponding authors

</div>

<p align="center">
  <img src="img/teaser.png" width="100%" alt="AtlasSegFM overview"/>
  <br>
  <em>AtlasSegFM customizes an off-the-shelf segmentation foundation model to a new clinical context with a single annotated atlas — no extra training set required.</em>
</p>

---

## News

- **[2026.07]** AtlasSegFM is accepted to **ECCV 2026** as a **Spotlight** paper! 🎉
- **[2026.08]** Official code and a one-shot demo (HaN-Seg) are released.

## Abstract

Accurate segmentation of anatomical structures in medical images is essential for diagnosis and treatment planning. While recent interactive segmentation foundation models enhance generalization through large-scale multimodal pretraining, they still depend on precise prompts and can fail in underrepresented clinical contexts (e.g., small organs-at-risk). We present **AtlasSegFM**, an atlas-guided framework that customizes off-the-shelf foundation models to new clinical contexts with a **single annotated example**. AtlasSegFM (1) performs atlas–query registration to generate context-aware prompts, (2) refines the segmentation with a frozen foundation model, and (3) applies a lightweight adaptive fusion module to combine atlas priors with foundation-model predictions. Extensive experiments on six public and in-house datasets across radiotherapy and vascular scenarios show consistent gains, with the largest improvements on small and delicate structures.

[[arXiv]](https://arxiv.org/abs/2512.18176)
[[PDF]](https://arxiv.org/pdf/2512.18176)
[[ECCV Spotlight]](https://eccv.ecva.net/virtual/2026/spotlight/6050)
[[ECCV Poster]](https://eccv.ecva.net/virtual/2026/poster/3573)

## Method

AtlasSegFM is an **inference-time** pipeline. The foundation model stays frozen; only a lightweight registration network and a fusion adapter are optimized per atlas–query pair.

| Step | Module | Role |
|------|--------|------|
| 1 | Atlas registration | Rigid + affine (ITK/Elastix) then deformable (RDP) to warp the support atlas into the query space and produce a context-aware mask prompt |
| 2 | Foundation-model refinement | Frozen [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) (or a compatible FM) refines the atlas prompt |
| 3 | Adaptive fusion | A lightweight 3D adapter fuses atlas priors with FM predictions; trained on the **support** case and applied to the **query** |

Registration always warps the **moving (support / atlas)** image and label into the **fixed (query)** space.

## Installation

A Linux machine with an NVIDIA GPU is recommended (used by Steps 1–3). Python 3.10+ is required.

```bash
git clone https://github.com/alfredtorres/AtlasSegFM.git
cd AtlasSegFM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch for your CUDA version first if needed: https://pytorch.org/get-started/locally/

### nnInteractive

```bash
pip install nnInteractive
```

Or from source:

```bash
git clone https://github.com/MIC-DKFZ/nnInteractive
cd nnInteractive
pip install -e ./client
pip install -e .
```

### Foundation-model weights

nnInteractive weights are **not** bundled (~400 MB). Download them from Hugging Face:

```bash
pip install huggingface_hub

huggingface-cli download MIC-DKFZ/nnInteractive \
  --include "nnInteractive_v1.0/*" \
  --local-dir models
```

After download, the following file should exist:

```
models/nnInteractive_v1.0/fold_0/checkpoint_final.pth
```

> Place the model at `models/nnInteractive_v1.0/` relative to the repository root. Step 2 uses this path by default (`--model-dir`).

nnInteractive model license: [CC BY-NC-SA 4.0](https://huggingface.co/MIC-DKFZ/nnInteractive) (non-commercial).

## Quick Start

This release includes a minimal pair from **[HaN-Seg](https://han-seg2023.grand-challenge.org/)** ([download](https://zenodo.org/records/7442914#.ZBtfBHbMJaQ); case **01** as query, case **06** as support), using the preprocessed CT volumes from our internal pipeline.

| Role | Volume | Registration | Label usage |
|------|--------|--------------|-------------|
| **Support** (moving / atlas) | `img_support` | Atlas with known segmentation | Used to train the fusion adapter |
| **Query** (fixed) | `img_query` | Target image to segment | **Query GT is used only for final Dice evaluation** |

Step 3 needs registration and FM results in **both** image spaces, so Steps 1 and 2 each run twice (query-fixed and support-fixed).

```bash
bash run_pipeline.sh
```

Or run the three stages manually:

```bash
# Step 1a: query — fixed=img_query, moving=img_support
python step1_registration.py

# Step 1b: support — fixed=img_support, moving=img_query
python step1_registration.py \
  --fixed-image test_data/images/img_support.nii.gz \
  --moving-image test_data/images/img_query.nii.gz \
  --fixed-label test_data/labels/label_support.nii.gz \
  --moving-label test_data/labels/label_query.nii.gz \
  --out-dir test_data/step1_output_support

# Step 2a: FM on query
python step2_FM_segment.py

# Step 2b: FM on support (Dice vs. label_fixed in registration space)
python step2_FM_segment.py \
  --reg-dir test_data/step1_output_support \
  --out-dir test_data/step2_output_support \
  --eval-dice

# Step 3: fusion (train on support, apply to query, report Dice on query)
python step3_fusion_adapter.py --device cuda:0 --full-res
```

Step 3 defaults to `cuda:3` at **full resolution** (~48 GB). Change `--device` as needed. If you run out of memory, omit `--full-res` or pass e.g. `--target-shape 128 128 128`.

## Custom Data

Point Step 1 at your own atlas–query pair (NIfTI):

```bash
python step1_registration.py \
  --fixed-image  /path/to/img_query.nii.gz \
  --moving-image /path/to/img_support.nii.gz \
  --fixed-label  /path/to/label_query.nii.gz \
  --moving-label /path/to/label_support.nii.gz \
  --out-dir      /path/to/output
```

Then run Step 2 on both registration directories and Step 3 with the corresponding `--query-*` / `--support-*` paths. Query labels are optional at inference and are used only when you want Dice evaluation.

## Repository Layout

```
AtlasSegFM_release_v1/
├── step1_registration.py      # rigid + affine + RDP
├── step2_FM_segment.py        # nnInteractive refinement
├── step3_fusion_adapter.py    # test-time fusion adapter
├── run_pipeline.sh
├── img/                       # teaser figure
├── model/                     # deformable registration network (RDP)
├── models/nnInteractive_v1.0/ # download separately (see above)
└── test_data/
    ├── images/
    │   ├── img_query.nii.gz       # query (fixed) CT  [HaN-Seg case 01]
    │   └── img_support.nii.gz     # support (moving) CT [HaN-Seg case 06]
    ├── labels/
    │   ├── label_query.nii.gz     # query GT (eval only)
    │   └── label_support.nii.gz   # support GT (adapter training)
    ├── step1_output/              # auto-generated (query registration)
    ├── step1_output_support/      # auto-generated (support registration)
    ├── step2_output/              # auto-generated (FM on query)
    ├── step2_output_support/      # auto-generated (FM on support)
    └── step3_output/              # auto-generated (fusion results)
```

Directories under `step1_output/` and below are created automatically by `bash run_pipeline.sh`.

## Outputs

| Step | Main output |
|------|-------------|
| 1 | `label_moved_final.nii.gz` — support label warped to fixed/query space |
| 2 | `FM_segment_results.nii.gz` — nnInteractive-refined segmentation |
| 3 | `query_fusion_results.nii.gz` — fused prediction + Dice on query GT (registration space) |

## Citation

If you use this code or find AtlasSegFM useful, please cite:

```bibtex
@inproceedings{zhang2026atlas,
  title     = {Atlas is Your Perfect Context: One-Shot Customization for Generalizable Foundational Medical Image Segmentation},
  author    = {Zhang, Ziyu and Yu, Yi and Zhu, Simeng and Aly, Ahmed and Gao, Yunhe and Gu, Ning and Xue, Yuan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

Please also cite **nnInteractive**, **VoxelMorph**, and **RDP** when using the corresponding components, and **HaN-Seg** if you use the bundled demo data.

## Acknowledgements

The deformable registration module in `model/` is adapted from existing open-source implementations. We thank the authors of the following projects:

- **[VoxelMorph](https://github.com/voxelmorph/voxelmorph)** — registration architecture and training utilities.
  Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019). VoxelMorph: a learning framework for deformable medical image registration. *IEEE TMI*, 38(8), 1788–1800.
- **[RDP](https://github.com/ZAX130/RDP)** — `model/model_rdp.py` is based on the RDP registration model shared by Haiqiao Wang (Shenzhen University), which itself builds upon VoxelMorph.
- **[nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)** — the default interactive foundation model in this release.
- **[HaN-Seg](https://han-seg2023.grand-challenge.org/)** ([download](https://zenodo.org/records/7442914#.ZBtfBHbMJaQ)) — the bundled `test_data/` volumes are derived from cases 01 (query) and 06 (support).

## Contact

For questions, please open a GitHub issue or contact the corresponding authors.
