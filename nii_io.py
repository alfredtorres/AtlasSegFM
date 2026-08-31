"""NIfTI I/O helpers — nibabel backend, sitk-compatible (z, y, x) array layout."""

import nibabel as nib
import numpy as np


def load_nii(path):
    return nib.load(path)


def load_array(path, as_label=False):
    """Load volume as numpy array in (z, y, x) order matching SimpleITK."""
    arr = nib.load(path).get_fdata()
    arr = arr.transpose(2, 1, 0)
    if as_label:
        arr = np.round(arr).astype(np.int32)
    return arr


def save_array(path, arr, ref_path, as_label=False):
    """Save (z, y, x) array to NIfTI using reference affine/header."""
    ref = nib.load(ref_path)
    out = arr.transpose(2, 1, 0)
    if as_label:
        out = np.round(out).astype(np.uint8)
    nib.save(nib.Nifti1Image(out, ref.affine, ref.header), path)
