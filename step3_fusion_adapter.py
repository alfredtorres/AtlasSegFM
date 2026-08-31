"""
Step 3: Per-label fusion adapter.

Trains a lightweight FusionAdapter on a support case and applies it to a query case,
fusing atlas-registration and FM segmentation predictions.
"""

import argparse
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from nii_io import load_array, save_array


def load_volume(path, as_label=False):
    arr = load_array(path, as_label=as_label)
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None, ...]
    return tensor, path


def resize_volume(tensor, target_shape, is_label=False):
    if target_shape is None:
        return tensor
    mode = "nearest" if is_label else "trilinear"
    if mode == "nearest":
        return F.interpolate(tensor, size=target_shape, mode=mode)
    return F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False)


def save_label_volume(tensor, ref_path, save_path):
    arr = tensor.detach().cpu().numpy().squeeze()
    save_array(save_path, arr, ref_path, as_label=True)


def soft_prob_from_binary(mask_tensor, tau=2.0):
    out = torch.zeros_like(mask_tensor)
    arr_np = mask_tensor.detach().cpu().numpy()
    for b in range(mask_tensor.shape[0]):
        m = arr_np[b, 0] > 0.5
        if m.any() and (~m).any():
            sdf = distance_transform_edt(m) - distance_transform_edt(~m)
        elif m.all():
            sdf = np.full_like(arr_np[b, 0], 10.0)
        else:
            sdf = np.full_like(arr_np[b, 0], -10.0)
        p = 1.0 / (1.0 + np.exp(-sdf / tau))
        out[b, 0] = torch.from_numpy(p).to(mask_tensor)
    return out.clamp(1e-4, 1 - 1e-4)


class FusionAdapter(nn.Module):
    def __init__(self, img_ch=1, base_ch=16):
        super().__init__()
        in_ch = img_ch + 6
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1),
            nn.InstanceNorm3d(base_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(base_ch, base_ch, 3, padding=1),
            nn.InstanceNorm3d(base_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(base_ch, base_ch, 3, padding=2, dilation=2),
            nn.InstanceNorm3d(base_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.gate_head = nn.Conv3d(base_ch, 1, 1)
        nn.init.kaiming_normal_(self.gate_head.weight, a=0.1)
        nn.init.zeros_(self.gate_head.bias)

    @staticmethod
    def _logit(p, eps=1e-4):
        p = p.clamp(eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)

    @staticmethod
    def _entropy(p, eps=1e-6):
        return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

    def forward(self, image, fm_soft, atlas_soft):
        zF = self._logit(fm_soft)
        zA = self._logit(atlas_soft)
        diff = zA - zF
        disagree = (fm_soft - atlas_soft).abs()
        hF = self._entropy(fm_soft)
        hA = self._entropy(atlas_soft)

        img = image - image.amin(dim=(2, 3, 4), keepdim=True)
        img = img / (img.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)

        feat = self.enc(torch.cat([img, zF, zA, diff, disagree, hF, hA], dim=1))
        gate_logit = self.gate_head(feat)
        g = torch.sigmoid(gate_logit)
        out = torch.sigmoid(zF + g * diff)
        return out, {"gate_logit": gate_logit, "disagree": disagree, "g": g}


def build_oracle_gate(fm_bin, atlas_bin, gt_bin):
    fm_correct = fm_bin == gt_bin
    atlas_correct = atlas_bin == gt_bin
    valid = fm_correct ^ atlas_correct
    return atlas_correct.float(), valid.float()


def fusion_loss(y_pred, y_true, aux, g_target, valid_mask):
    eps = 1e-6
    y_pred_flat = y_pred.contiguous().view(-1)
    y_true_flat = y_true.contiguous().view(-1)
    intersection = (y_pred_flat * y_true_flat).sum()
    l_dice = 1.0 - (2.0 * intersection + eps) / (y_pred_flat.sum() + y_true_flat.sum() + eps)

    bce_each = F.binary_cross_entropy_with_logits(aux["gate_logit"], g_target, reduction="none")
    l_gate_sup = (bce_each * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    w_map = (aux["disagree"].detach() > 0.1).float()
    if w_map.sum() > 0:
        l_bce_d = (
            F.binary_cross_entropy(y_pred.clamp(1e-6, 1 - 1e-6), y_true, reduction="none") * w_map
        ).sum() / (w_map.sum() + 1e-6)
    else:
        l_bce_d = torch.tensor(0.0, device=y_pred.device)

    return l_dice + 5.0 * l_gate_sup + l_bce_d


def get_valid_labels(*tensors):
    labels = set()
    for tensor in tensors:
        labels.update(np.unique(tensor.numpy().astype(np.int32)).tolist())
    return sorted(lb for lb in labels if lb != 0)


def train_label_adapter(
    support_image,
    query_image,
    support_gt,
    fm_support,
    atlas_support,
    fm_query,
    atlas_query,
    num_epochs,
    device,
):
    fm_support_soft = soft_prob_from_binary(fm_support, tau=2.0)
    atlas_support_soft = soft_prob_from_binary(atlas_support, tau=2.0)
    fm_query_soft = soft_prob_from_binary(fm_query, tau=2.0)
    atlas_query_soft = soft_prob_from_binary(atlas_query, tau=2.0)
    g_target, valid_mask = build_oracle_gate(fm_support, atlas_support, support_gt)

    adapter = FusionAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.3)

    query_prob = None
    for _ in range(num_epochs):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        y_pred, aux = adapter(support_image, fm_support_soft, atlas_support_soft)
        loss = fusion_loss(y_pred, support_gt, aux, g_target, valid_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            adapter.eval()
            query_prob, _ = adapter(query_image, fm_query_soft, atlas_query_soft)

    return query_prob.detach().cpu()


def compute_multilabel_dice(gt_raw, pred_raw):
    gt_np = gt_raw.numpy().astype(np.int32).ravel()
    pred_np = pred_raw.numpy().astype(np.int32).ravel()
    labels = sorted(set(gt_np.tolist()) | set(pred_np.tolist()))
    labels = [lb for lb in labels if lb != 0]
    dices = []
    for lb in labels:
        gt_bin = (gt_raw == lb).numpy()
        pred_bin = (pred_raw == lb).numpy()
        intersection = np.logical_and(gt_bin, pred_bin).sum()
        union = gt_bin.sum() + pred_bin.sum()
        dices.append((lb, (2.0 * intersection) / union if union > 0 else 1.0))
    macro = float(np.mean([d for _, d in dices])) if dices else 0.0
    return dices, macro


def run_fusion(
    query_image_path,
    query_reg_dir,
    support_image_path,
    support_label_path,
    atlas_query_path,
    atlas_support_path,
    fm_query_path,
    fm_support_path,
    out_dir,
    num_epochs=30,
    target_shape=(128, 128, 128),
    device=None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        raise RuntimeError("Fusion adapter training requires a CUDA GPU.")

    query_gt_path = os.path.join(query_reg_dir, "label_fixed.nii.gz")

    for path in [
        query_image_path,
        query_gt_path,
        support_image_path,
        support_label_path,
        atlas_query_path,
        atlas_support_path,
        fm_query_path,
        fm_support_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    query_image, _ = load_volume(query_image_path)
    query_gt_raw, query_gt_path = load_volume(query_gt_path, as_label=True)
    support_image, _ = load_volume(support_image_path)
    support_gt_raw, _ = load_volume(support_label_path, as_label=True)
    atlas_query_raw, _ = load_volume(atlas_query_path, as_label=True)
    atlas_support_raw, _ = load_volume(atlas_support_path, as_label=True)
    fm_query_raw, _ = load_volume(fm_query_path, as_label=True)
    fm_support_raw, _ = load_volume(fm_support_path, as_label=True)

    orig_query_shape = tuple(query_gt_raw.shape[2:])
    query_image = resize_volume(query_image, target_shape, is_label=False)
    query_gt_raw = resize_volume(query_gt_raw, target_shape, is_label=True)
    support_image = resize_volume(support_image, target_shape, is_label=False)
    support_gt_raw = resize_volume(support_gt_raw, target_shape, is_label=True)
    atlas_query_raw = resize_volume(atlas_query_raw, target_shape, is_label=True)
    atlas_support_raw = resize_volume(atlas_support_raw, target_shape, is_label=True)
    fm_query_raw = resize_volume(fm_query_raw, target_shape, is_label=True)
    fm_support_raw = resize_volume(fm_support_raw, target_shape, is_label=True)

    query_image = query_image.to(device)
    support_image = support_image.to(device)

    if target_shape:
        print(f"[Step 3] Training at {target_shape}, original query shape {orig_query_shape}")
    else:
        print(f"[Step 3] Training at full resolution {orig_query_shape}")

    valid_labels = get_valid_labels(support_gt_raw, atlas_support_raw, fm_support_raw)
    print(f"[Step 3] Fusing {len(valid_labels)} labels ...")

    prob_threshold = 0.5
    query_best_prob = torch.full_like(query_gt_raw, prob_threshold)
    query_pred = torch.zeros_like(query_gt_raw)

    for label_id in valid_labels:
        support_gt = (support_gt_raw == label_id).float()
        if support_gt.sum() == 0:
            continue

        query_prob = train_label_adapter(
            support_image=support_image,
            query_image=query_image,
            support_gt=support_gt.to(device),
            fm_support=(fm_support_raw == label_id).float().to(device),
            atlas_support=(atlas_support_raw == label_id).float().to(device),
            fm_query=(fm_query_raw == label_id).float().to(device),
            atlas_query=(atlas_query_raw == label_id).float().to(device),
            num_epochs=num_epochs,
            device=device,
        )

        update = query_prob > query_best_prob
        query_best_prob = torch.where(update, query_prob, query_best_prob)
        query_pred = torch.where(
            update,
            torch.full_like(query_pred, float(label_id)),
            query_pred,
        )
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(out_dir, "query_fusion_results.nii.gz")
    if target_shape:
        query_pred_full = resize_volume(query_pred, orig_query_shape, is_label=True)
    else:
        query_pred_full = query_pred
    save_label_volume(query_pred_full, query_gt_path, out_path)
    print(f"[Step 3] Saved: {out_path}")

    label_dices, macro_dice = compute_multilabel_dice(query_gt_raw, query_pred_full)
    print("[Step 3] ------- Final Dice (query GT in registration space) -------")
    for lb, dice in label_dices:
        print(f"  Label {lb:2d}: Dice = {dice:.4f}")
    print(f"[Step 3] Mean Dice = {macro_dice:.4f}")


def parse_args():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Step 3: fusion adapter")

    # Support = moving image, label available for adapter training
    parser.add_argument(
        "--support-image",
        default=os.path.join(root, "test_data/images/img_support.nii.gz"),
    )
    parser.add_argument(
        "--support-label",
        default=os.path.join(root, "test_data/labels/label_support.nii.gz"),
    )
    parser.add_argument(
        "--support-reg-dir",
        default=os.path.join(root, "test_data/step1_output_support"),
    )
    parser.add_argument(
        "--support-fm-dir",
        default=os.path.join(root, "test_data/step2_output_support"),
    )

    # Query = fixed image, label used for final Dice evaluation only
    parser.add_argument(
        "--query-image",
        default=os.path.join(root, "test_data/images/img_query.nii.gz"),
    )
    parser.add_argument(
        "--query-reg-dir",
        default=os.path.join(root, "test_data/step1_output"),
        help="Step1 query outputs; label_fixed.nii.gz is used for final Dice",
    )
    parser.add_argument(
        "--query-fm-dir",
        default=os.path.join(root, "test_data/step2_output"),
    )
    parser.add_argument("--out-dir", default=os.path.join(root, "test_data/step3_output"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--full-res",
        action="store_true",
        help="Train at native image resolution (requires ~48 GB GPU memory)",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Downsampled training size (default: 128 128 128, ignored if --full-res)",
    )
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full_res:
        target_shape = None
    elif args.target_shape is not None:
        target_shape = tuple(args.target_shape)
    else:
        target_shape = (128, 128, 128)
    run_fusion(
        query_image_path=args.query_image,
        query_reg_dir=args.query_reg_dir,
        support_image_path=args.support_image,
        support_label_path=args.support_label,
        atlas_query_path=os.path.join(args.query_reg_dir, "label_moved_final.nii.gz"),
        atlas_support_path=os.path.join(args.support_reg_dir, "label_moved_final.nii.gz"),
        fm_query_path=os.path.join(args.query_fm_dir, "FM_segment_results.nii.gz"),
        fm_support_path=os.path.join(args.support_fm_dir, "FM_segment_results.nii.gz"),
        out_dir=args.out_dir,
        num_epochs=args.epochs,
        target_shape=target_shape,
        device=args.device,
    )
