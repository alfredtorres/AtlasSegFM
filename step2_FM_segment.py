"""
Step 2: Foundation-model interactive segmentation (nnInteractive).

Refines the atlas registration result on the fixed image using nnInteractive.
All volume I/O uses nibabel for consistent orientation with step 1.
"""

import argparse
import os

import numpy as np
import torch
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from nii_io import load_array, save_array


def compute_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    return (2.0 * intersection) / union if union > 0 else 1.0


def fm_segment(reg_dir, out_dir, model_dir, label_list, device, eval_dice=False):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"nnInteractive model not found at: {model_dir}\n"
            "Please download the model and place it as described in README.md."
        )

    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(model_dir)

    image_path = os.path.join(reg_dir, "image_fixed_norm.nii.gz")
    img = load_array(image_path, as_label=False).astype(np.float32)

    prev_pred = load_array(os.path.join(reg_dir, "label_moved_final.nii.gz"), as_label=True)

    label_data = None
    if eval_dice:
        label_data = load_array(os.path.join(reg_dir, "label_fixed.nii.gz"), as_label=True)

    active_labels = [lb for lb in label_list if (prev_pred == lb).any()]
    if not active_labels:
        active_labels = sorted(int(x) for x in np.unique(prev_pred) if x > 0)

    session.set_image(img[None].astype(np.float32))
    target_buffer = torch.zeros(img.shape, dtype=torch.uint8, device="cpu")
    session.set_target_buffer(target_buffer)

    result_mask = np.zeros(img.shape, dtype=np.uint8)
    dice_atlas = []
    dice_fm = []

    print(f"[Step 2] FM segmentation on {len(active_labels)} labels ...")
    for label in active_labels:
        print(f"[Step 2] Label {label}")
        session.reset_interactions()
        session.add_initial_seg_interaction(
            (prev_pred == label).astype(np.uint8),
            run_prediction=True,
        )
        results = session.target_buffer.clone().cpu().numpy()
        result_mask[results == 1] = label

        if eval_dice and label_data is not None:
            gt = (label_data == label).astype(np.uint8)
            pred_atlas = (prev_pred == label).astype(np.uint8)
            pred_fm = (results == 1).astype(np.uint8)
            dice_atlas.append(compute_dice(gt, pred_atlas))
            dice_fm.append(compute_dice(gt, pred_fm))

    out_path = os.path.join(out_dir, "FM_segment_results.nii.gz")
    save_array(out_path, result_mask, image_path, as_label=True)

    if eval_dice and dice_atlas:
        dice_atlas = np.array(dice_atlas)
        dice_fm = np.array(dice_fm)
        print("[Step 2] ------- Dice (GT in registration space) -------")
        for label, d_atlas, d_fm in zip(active_labels, dice_atlas, dice_fm):
            print(f"  Label {label:2d}: Dice (atlas) = {d_atlas:.4f}, Dice (FM) = {d_fm:.4f}")
        print(f"[Step 2] Mean Dice (atlas) = {dice_atlas.mean():.4f}")
        print(f"[Step 2] Mean Dice (FM) = {dice_fm.mean():.4f}")

    print(f"[Step 2] Saved: {out_path}")


def parse_args():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Step 2: FM interactive segmentation")
    parser.add_argument(
        "--reg-dir",
        default=os.path.join(root, "test_data/step1_output"),
        help="Directory containing step1 registration outputs",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(root, "test_data/step2_output"),
        help="Output directory for step2 results",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(root, "models/nnInteractive_v1.0"),
        help="Path to nnInteractive_v1.0 model folder",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=list(range(1, 31)),
        help="Label IDs to refine",
    )
    parser.add_argument(
        "--eval-dice",
        action="store_true",
        help="Compute Dice against label_fixed.nii.gz in reg-dir",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for nnInteractive",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fm_segment(
        reg_dir=args.reg_dir,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        label_list=args.labels,
        device=args.device,
        eval_dice=args.eval_dice,
    )
