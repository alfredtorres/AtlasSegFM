"""
Step 1: Atlas registration pipeline.

Combines rigid + affine registration (ITK/Elastix) and deformable registration (RDP).
Volume I/O for RDP uses nibabel consistently with the original pipeline.
"""

import argparse
import json
import os

import itk
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from model.loss import gradient_loss, mse_loss
from model.model_rdp import RDP, register_model
from nii_io import load_array, save_array


# ---------------------------------------------------------------------------
# Preprocessing helpers (SimpleITK for Elastix only)
# ---------------------------------------------------------------------------
def reorient_image(image, desired_orientation="LPI"):
    current_orientation = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(
        image.GetDirection()
    )
    if current_orientation != desired_orientation:
        reorient_filter = sitk.DICOMOrientImageFilter()
        reorient_filter.SetDesiredCoordinateOrientation(desired_orientation)
        image = reorient_filter.Execute(image)
    return image


def normalize_img(img):
    arr = sitk.GetArrayViewFromImage(img)
    result = sitk.GetImageFromArray(arr)
    result.CopyInformation(img)
    return result


def sitk_label_from_reoriented(label_img):
    arr = sitk.GetArrayViewFromImage(label_img)
    result = sitk.GetImageFromArray(arr)
    result.CopyInformation(label_img)
    return result


def maybe_reorient(image, orientation):
    if orientation:
        return reorient_image(image, orientation)
    return image


# ---------------------------------------------------------------------------
# Rigid + affine registration
# ---------------------------------------------------------------------------
def rigid_affine_registration(
    moving_path, fixed_path, moving_label_path, fixed_label_path, out_dir, orientation=None
):
    os.makedirs(out_dir, exist_ok=True)

    image_fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    image_fixed_norm = normalize_img(maybe_reorient(image_fixed, orientation))
    sitk.WriteImage(image_fixed_norm, os.path.join(out_dir, "image_fixed_norm.nii.gz"))

    image_moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    image_moving_norm = normalize_img(maybe_reorient(image_moving, orientation))
    sitk.WriteImage(image_moving_norm, os.path.join(out_dir, "image_moving_norm.nii.gz"))

    label_fixed = sitk_label_from_reoriented(
        maybe_reorient(sitk.ReadImage(fixed_label_path, sitk.sitkUInt8), orientation)
    )
    sitk.WriteImage(label_fixed, os.path.join(out_dir, "label_fixed.nii.gz"))

    label_moving = sitk_label_from_reoriented(
        maybe_reorient(sitk.ReadImage(moving_label_path, sitk.sitkUInt8), orientation)
    )
    sitk.WriteImage(label_moving, os.path.join(out_dir, "label_moving.nii.gz"))

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(itk.ParameterObject.GetDefaultParameterMap("rigid"))
    parameter_object.AddParameterMap(itk.ParameterObject.GetDefaultParameterMap("affine"))

    fixed_path_norm = os.path.join(out_dir, "image_fixed_norm.nii.gz")
    moving_path_norm = os.path.join(out_dir, "image_moving_norm.nii.gz")

    elx = itk.ElastixRegistrationMethod.New(itk.imread(fixed_path_norm), itk.imread(moving_path_norm))
    elx.SetParameterObject(parameter_object)
    elx.SetLogToConsole(False)
    elx.SetLogToFile(True)
    elx.SetLogFileName(os.path.join(out_dir, "elastix.log"))
    elx.SetOutputDirectory(out_dir)

    print("[Step 1] Rigid + affine registration ...")
    elx.UpdateLargestPossibleRegion()
    result_tpobj = elx.GetTransformParameterObject()

    reg_img_path = os.path.join(out_dir, "result_image.mha")
    itk.imwrite(elx.GetOutput(), reg_img_path)

    tp_files = []
    for i in range(result_tpobj.GetNumberOfParameterMaps()):
        fn = os.path.join(out_dir, f"TransformParameters.{i}.txt")
        itk.ParameterObject.New().WriteParameterFile(result_tpobj.GetParameterMap(i), fn)
        tp_files.append(fn)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "fixed": fixed_path_norm,
                "moving": moving_path_norm,
                "result_image": reg_img_path,
                "parameter_files": tp_files,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(os.path.join(out_dir, "TransformParameters.0.txt"))
    parameter_object.AddParameterFile(os.path.join(out_dir, "TransformParameters.1.txt"))

    moved_image = itk.transformix_filter(
        itk.imread(os.path.join(out_dir, "image_moving_norm.nii.gz"), itk.F),
        parameter_object,
    )
    itk.imwrite(moved_image.astype(itk.F), os.path.join(out_dir, "image_moved.nii.gz"))

    parameter_object.SetParameter(0, "FinalBSplineInterpolationOrder", "0")
    parameter_object.SetParameter(1, "FinalBSplineInterpolationOrder", "0")
    moved_label = itk.transformix_filter(
        itk.imread(os.path.join(out_dir, "label_moving.nii.gz"), itk.UC),
        parameter_object,
    )
    itk.imwrite(moved_label.astype(itk.UC), os.path.join(out_dir, "label_moved.nii.gz"))

    diff = sitk.Square(
        sitk.Subtract(sitk.ReadImage(fixed_path_norm), sitk.ReadImage(reg_img_path))
    )
    mse = float(sitk.GetArrayViewFromImage(diff).mean())
    print(f"[Step 1] Affine registration MSE = {mse:.6f}")


# ---------------------------------------------------------------------------
# RDP deformable registration (nibabel I/O)
# ---------------------------------------------------------------------------
def load_nii_to_tensor(path, device=None, mode="data", target_shape=(128, 128, 128)):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arr = load_array(path, as_label=(mode == "label"))
    x = torch.from_numpy(arr.astype(np.float32))[None, None, ...].to(device=device)
    if mode == "data":
        x = F.interpolate(x, target_shape, mode="trilinear", align_corners=False)
    else:
        x = F.interpolate(x, target_shape, mode="nearest")
    return x


def rdp_registration(step1_dir, out_dir, max_epochs=10, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        raise RuntimeError("RDP registration requires a CUDA GPU.")

    img_moving = load_nii_to_tensor(
        os.path.join(step1_dir, "image_moved.nii.gz"), device=device, mode="data"
    )
    img_fixed = load_nii_to_tensor(
        os.path.join(step1_dir, "image_fixed_norm.nii.gz"), device=device, mode="data"
    )
    label_moving = load_nii_to_tensor(
        os.path.join(step1_dir, "label_moved.nii.gz"), device=device, mode="label"
    )

    vol_size = img_moving.shape[2:]
    model = RDP(vol_size, channels=16).to(device)
    reg_model = register_model(vol_size, "nearest").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = PolynomialLR(optimizer, total_iters=max_epochs, power=0.9)

    print(f"[Step 1] RDP deformable registration ({max_epochs} epochs on {device}) ...")
    label_moved = None
    for epoch in range(max_epochs):
        model.train()
        for _ in range(100):
            y_moved, flow = model(img_moving, img_fixed)
            loss = mse_loss(y_moved, img_fixed) + 0.01 * gradient_loss(flow)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            y_move, flow_m2f = model(img_moving, img_fixed)
            label_moved = reg_model([label_moving.float(), flow_m2f])
            img_error = torch.mean((img_fixed - y_move) ** 2)
            print(f"[Step 1] RDP epoch {epoch}: image MSE = {img_error.item():.6f}")

    fixed_label_path = os.path.join(step1_dir, "label_fixed.nii.gz")
    target_shape = load_array(fixed_label_path, as_label=True).shape
    label_moved = F.interpolate(label_moved, target_shape, mode="nearest")

    out_path = os.path.join(out_dir, "label_moved_final.nii.gz")
    moved_arr = np.round(label_moved.cpu().numpy().squeeze()).astype(np.uint8)
    save_array(out_path, moved_arr, fixed_label_path, as_label=True)
    print(f"[Step 1] Saved: {out_path}")


def run_registration(
    moving_image,
    fixed_image,
    moving_label,
    fixed_label,
    out_dir,
    max_epochs=10,
    device=None,
    orientation=None,
):
    os.makedirs(out_dir, exist_ok=True)
    rigid_affine_registration(
        moving_image, fixed_image, moving_label, fixed_label, out_dir, orientation=orientation
    )
    rdp_registration(out_dir, out_dir, max_epochs=max_epochs, device=device)
    print(f"[Step 1] Registration outputs saved to: {out_dir}")


def parse_args():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Step 1: atlas registration (affine + RDP)")
    parser.add_argument(
        "--fixed-image",
        default=os.path.join(root, "test_data/images/img_query.nii.gz"),
        help="Fixed (query/test) image — segmentation target",
    )
    parser.add_argument(
        "--moving-image",
        default=os.path.join(root, "test_data/images/img_support.nii.gz"),
        help="Moving (support) image — atlas with available label",
    )
    parser.add_argument(
        "--fixed-label",
        default=os.path.join(root, "test_data/labels/label_query.nii.gz"),
        help="Label of the fixed image (query GT, for evaluation only in the demo)",
    )
    parser.add_argument(
        "--moving-label",
        default=os.path.join(root, "test_data/labels/label_support.nii.gz"),
        help="Label of the moving image (support GT, used for training)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(root, "test_data/step1_output"),
        help="Output directory",
    )
    parser.add_argument(
        "--reorient",
        default=None,
        help="Optional DICOM orientation code (e.g. LPI). "
        "HaN-Seg preprocessed data should leave this unset.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="RDP training epochs")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for RDP",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_registration(
        moving_image=args.moving_image,
        fixed_image=args.fixed_image,
        moving_label=args.moving_label,
        fixed_label=args.fixed_label,
        out_dir=args.out_dir,
        max_epochs=args.epochs,
        device=args.device,
        orientation=args.reorient,
    )
