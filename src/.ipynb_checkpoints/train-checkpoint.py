from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import tqdm
from torch.nn import L1Loss
import visdom
import nibabel as nib
import numpy as np
import os
from monai.utils import first, set_determinism
from torch.optim import Adam


import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
from monai.metrics import compute_hausdorff_distance
from functools import partial
# from monai.inferers import sliding_window_inference

# from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import math
import torch
import torch.nn.functional as F

from monai.transforms import CenterSpatialCrop
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
)
# from monai.losses import DiceCELoss
# import matplotlib
# matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you're using Qt
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
from torch.cuda.amp import autocast
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from src.custom_sliding_window import sliding_window_custom_collect
wandb.init(project="MAMBA_UNET", name="Mamba_Unet_test_run")

# matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you're using Qt
from tqdm import tqdm

# tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact"

__all__ = ["sliding_window_inference"]


def sliding_window_inference(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        process_fn: process inference output and adjust the importance map per window
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.
        with_coord: whether to pass the window coordinates to ``predictor``. Default is False.
            If True, the signature of ``predictor`` should be ``predictor(patch_data, patch_coord, ...)``.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    buffered = buffer_steps is not None and buffer_steps > 0
    num_spatial_dims = len(inputs.shape) - 2
    if buffered:
        if buffer_dim < -num_spatial_dims or buffer_dim > num_spatial_dims:
            raise ValueError(f"buffer_dim must be in [{-num_spatial_dims}, {num_spatial_dims}], got {buffer_dim}.")
        if buffer_dim < 0:
            buffer_dim += num_spatial_dims
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype
    print("sliding window of train is being used now")
    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=not buffered)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    if not buffered:
        non_blocking = False
        windows_range = range(0, total_slices, sw_batch_size)
    else:
        slices, n_per_batch, b_slices, windows_range = _create_buffered_slices(
            slices, batch_size, sw_batch_size, buffer_dim, buffer_steps
        )
        non_blocking, _ss = torch.cuda.is_available(), -1
        for x in b_slices[:n_per_batch]:
            if x[1] < _ss:  # detect overlapping slices
                non_blocking = False
                break
            _ss = x[2]

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map_ = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map_ = compute_importance_map(
                valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
            )
            if len(importance_map_.shape) == num_spatial_dims and not process_fn:
                importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in tqdm(windows_range) if progress else windows_range:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, b_slices[b_s][0] if buffered else total_slices))
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
        if with_coord:
            seg_prob_out = predictor(win_data, unravel_slice, *args, **kwargs)  # batched patch
        else:
            seg_prob_out = predictor(win_data, *args, **kwargs)  # batched patch

        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        if process_fn:
            seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
        else:
            w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        if buffered:
            c_start, c_end = b_slices[b_s][1:]
            if not sw_device_buffer:
                k = seg_tuple[0].shape[1]  # len(seg_tuple) > 1 is currently ignored
                sp_size = list(image_size)
                sp_size[buffer_dim] = c_end - c_start
                sw_device_buffer = [torch.zeros(size=[1, k, *sp_size], dtype=compute_dtype, device=sw_device)]
            for p, s in zip(seg_tuple[0], unravel_slice):
                offset = s[buffer_dim + 2].start - c_start
                s[buffer_dim + 2] = slice(offset, offset + roi_size[buffer_dim])
                s[0] = slice(0, 1)
                sw_device_buffer[0][s] += p * w_t
            b_i += len(unravel_slice)
            if b_i < b_slices[b_s][0]:
                continue
        else:
            sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if not buffered and seg_shape != roi_size:
                z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor: Callable = torch.empty if non_blocking else torch.zeros  # type: ignore
                output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                w_t_ = w_t.to(device)
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_
            if buffered:
                o_slice = [slice(None)] * len(inputs.shape)
                o_slice[buffer_dim + 2] = slice(c_start, c_end)
                img_b = b_s // n_per_batch  # image batch index
                o_slice[0] = slice(img_b, img_b + 1)
                if non_blocking:
                    output_image_list[0][o_slice].copy_(sw_device_buffer[0], non_blocking=non_blocking)
                else:
                    output_image_list[0][o_slice] += sw_device_buffer[0].to(device=device)
            else:
                sw_device_buffer[ss] *= w_t
                sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
                _compute_coords(unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])
        sw_device_buffer = []
        if buffered:
            b_s += 1

    if non_blocking:
        torch.cuda.current_stream().synchronize()

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        kwargs.update({"pad_size": pad_size})
        for ss, output_i in enumerate(output_image_list):
            zoom_scale = [_shape_d / _roi_size_d for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)]
            final_slicing: list[slice] = []
            for sp in range(num_spatial_dims):
                si = num_spatial_dims - sp - 1
                slice_dim = slice(
                    int(round(pad_size[sp * 2] * zoom_scale[si])),
                    int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                )
                final_slicing.insert(0, slice_dim)
            output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

    return final_output  # type: ignore


def _create_buffered_slices(slices, batch_size, sw_batch_size, buffer_dim, buffer_steps):
    """rearrange slices for buffering"""
    slices_np = np.asarray(slices)
    slices_np = slices_np[np.argsort(slices_np[:, buffer_dim, 0], kind="mergesort")]
    slices = [tuple(slice(c[0], c[1]) for c in i) for i in slices_np]
    slices_np = slices_np[:, buffer_dim]

    _, _, _b_lens = np.unique(slices_np[:, 0], return_counts=True, return_index=True)
    b_ends = np.cumsum(_b_lens).tolist()  # possible buffer flush boundaries
    x = [0, *b_ends][:: min(len(b_ends), int(buffer_steps))]
    if x[-1] < b_ends[-1]:
        x.append(b_ends[-1])
    n_per_batch = len(x) - 1
    windows_range = [
        range(b * x[-1] + x[i], b * x[-1] + x[i + 1], sw_batch_size)
        for b in range(batch_size)
        for i in range(n_per_batch)
    ]
    b_slices = []
    for _s, _r in enumerate(windows_range):
        s_s = slices_np[windows_range[_s - 1].stop % len(slices) if _s > 0 else 0, 0]
        s_e = slices_np[(_r.stop - 1) % len(slices), 1]
        b_slices.append((_r.stop, s_s, s_e))  # buffer index, slice start, slice end
    windows_range = itertools.chain(*windows_range)  # type: ignore
    return slices, n_per_batch, b_slices, windows_range


def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[idx_zm] += p


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)
    
mse_loss = L1Loss()
class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.NONE,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                # print("target shape is", target.shape)
                target = one_hot(target, num_classes=n_pred_ch)
                # print("target shape is", target.shape)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                print("target shape is", target.shape)
                print("input shape is", input.shape)

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o
        # print(f"Intersection: {intersection}")
        # print(f"Denominator: {denominator}")

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        dice = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        # print(f"Dice: {dice}")

        num_of_classes = target.shape[1]
        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            f = f * self.class_weight.to(f)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


def masked_per_sample_dice_loss(pred, target, class_weights=None, epsilon=1e-5):
    """
    pred, target: [B, C, D, H, W]
    class_weights: tensor of shape [C]
    """
    B, C = pred.shape[:2]
    total_loss = []

    for b in range(B):
        sample_losses = []
        for c in range(C):
            p = pred[b, c]
            t = target[b, c]

            if torch.sum(t) == 0:
                continue  # Skip class not present

            intersection = torch.sum(p * t)
            denominator = torch.sum(p) + torch.sum(t)
            dice = (2 * intersection + epsilon) / (denominator + epsilon)
            loss = 1 - dice

            print("soft loss is", loss)

            if class_weights is not None:
                loss *= class_weights[c]

            sample_losses.append(loss)

        if sample_losses:
            total_loss.append(torch.mean(torch.stack(sample_losses)))

    if len(total_loss) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.mean(torch.stack(total_loss))





class DiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss`` and ``torch.nn.BCEWithLogitsLoss()``.
    In this implementation, two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    @deprecated_arg(
        "ce_weight", since="1.2", removed="1.4", new_name="weight", msg_suffix="please use `weight` instead."
    )
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: a rescaling weight given to each class for cross entropy loss for `CrossEntropyLoss`.
                or a weight of positive examples to be broadcasted with target used as `pos_weight` for `BCEWithLogitsLoss`.
                See ``torch.nn.CrossEntropyLoss()`` or ``torch.nn.BCEWithLogitsLoss()`` for more information.
                The weight is also used in `DiceLoss`.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        weight = ce_weight if ce_weight is not None else weight
        dice_weight: torch.Tensor | None
        if weight is not None and not include_background:
            dice_weight = weight[1:]
        else:
            dice_weight = weight
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=dice_weight,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(pos_weight=weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        
        target = torch.argmax(target, dim=1)
        # print("target shape is", target.shape)
        # print("torch.unique labesl are", torch.unique(target))
        cross_ent_loss = self.cross_entropy(input, target)  # type: ignore[no-any-return]
        # print("cross_ent_loss", cross_ent_loss)
        return cross_ent_loss

    def bce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary CrossEntropy loss for the input logits and target in one single class.

        """
        if not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.binary_cross_entropy(input, target)  # type: ignore[no-any-return]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.

        """
        # print("now dice ces is custom coded")
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )


        # dice_loss = masked_per_sample_dice_loss(input, target)


        
        dice_loss = self.dice(input, target)
        # print("target shape is", target.shape)
        ce_loss = self.ce(input, target) if input.shape[1] != 1 else self.bce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss


def compute_per_sample_volume_weights(y_onehot):
    """
    y_onehot: [B, 4, D, H, W]
    returns: [B, 4] tensor of weights for each sample
    """
    B, C, D, H, W = y_onehot.shape
    total_voxels = D * H * W

    weights = []

    for i in range(B):
        sample = y_onehot[i]  # shape: [4, D, H, W]
        class_voxel_counts = torch.sum(sample, dim=(1, 2, 3))  # shape: [4]
        class_ratios = class_voxel_counts / total_voxels

        bg_weight = 1.0
        nc_weight = 1.0 + class_ratios[1]
        ed_weight = 1.0
        et_weight = 1.0 + (1.0 - class_ratios[3]) * 2.0

        weights.append([bg_weight, nc_weight, ed_weight, et_weight])

    return torch.tensor(weights, device=y_onehot.device)  # shape: [B, 4]





weight_BG = 1.0   # Weight for Edema class
weight_ED = 1.0   # Weight for Edema class
weight_NC = 2.0   # Weight for Necrotic Core class (higher because it's underperforming)
weight_ET = 2.0   # Weight for Enhancing Tumor class (higher because it's underperforming)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([weight_BG, weight_NC, weight_ED, weight_ET], dtype=torch.float32).to(device)
dice_loss = DiceLoss(to_onehot_y=False, softmax=False)
dice_loss2 = DiceLoss(to_onehot_y=False, softmax=False)
loss_function = DiceCELoss(to_onehot_y=False, softmax=False, include_background=True, lambda_dice=0.8, lambda_ce=0.2)   

# weight for CE)

# dice_loss = DiceLoss(to_onehot_y=True, softmax=False)

# weights = compute_per_sample_volume_weights(y_onehot)

scaler = GradScaler()








def train_vae(model, train_loader, train_dataset_len, optimizer, device, epoch, global_step):
    """
    Train the VAE model for one epoch with mixed precision.
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()
    epoch_loss = 0
    
    

    class_losses_sum_overall_wo = {'BG': 0, 'NC': 0, 'ED': 0, 'ET': 0}
    class_losses_sum_overall = {'BG': 0, 'TC': 0, 'WT': 0, 'ET': 0}
    class_counts = {k: 0 for k in [0, 1, 2]}

    torch.autograd.set_detect_anomaly(True)

    for batch in tqdm(train_loader, desc=f"Training Secondary Network (Epoch {epoch})"):
        images = {key: batch[key] for key in ["t1n", "t2w", "t1c", "t2f"] if key in batch}
        # print("[images['t1n']", images['t1n'].shape)
        # print("[images['t2w']", images['t2w'].shape)
        # print("[images['t1c']", images['t1c'].shape)
        # print("[images['t2f']", images['t2f'].shape)
        
        images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1).to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            
            reconstruction, quantized_loss, latent, feature = model(images, mode='train', perform_retrieval=True)
            ce_dice_loss = loss_function(reconstruction, mask)
            combined_loss = dice_loss(reconstruction, mask).mean(dim=0)

            # print(f"BG_Loss______{combined_loss[0]}____________NC_Loss_________{combined_loss[1]}_____________ED_loss___________{combined_loss[2]}__________ET_Loss_________{combined_loss[3]}")

            batch_images = mask.shape[0]
            for idx, key in enumerate(class_losses_sum_overall_wo):
                class_losses_sum_overall_wo[key] += combined_loss[idx].item() * batch_images

            mask = torch.argmax(mask, dim=1)
            
            mask = torch.stack([
                (mask == 0),
                (mask == 1) | (mask == 3),
                (mask == 1) | (mask == 3) | (mask == 2),
                (mask == 3)
            ], dim=1).float()

            reconstruction = torch.argmax(reconstruction, dim=1)
            reconstruction = torch.stack([
                (reconstruction == 0),
                (reconstruction == 1) | (reconstruction == 3),
                (reconstruction == 1) | (reconstruction == 3) | (reconstruction == 2),
                (reconstruction == 3)
            ], dim=1).float()

            combined_loss_bts = dice_loss(reconstruction, mask).mean(dim=0)
            for idx, key in enumerate(class_losses_sum_overall):
                class_losses_sum_overall[key] += combined_loss_bts[idx].item() * batch_images

            loss = ce_dice_loss + quantized_loss
            loss_tr = loss * batch_images

        scaler.scale(loss).backward()


        scaler.step(optimizer)
        scaler.update()
        # global_step += 1

        epoch_loss += loss_tr.item()

    for key in class_losses_sum_overall_wo:
        class_losses_sum_overall_wo[key] /= train_dataset_len

    avaerage_dice_train = 0
    for key in class_losses_sum_overall:
        class_losses_sum_overall[key] /= train_dataset_len
        avaerage_dice_train += class_losses_sum_overall[key]
    avaerage_dice_train /= 4

    return epoch_loss / train_dataset_len, class_losses_sum_overall, class_losses_sum_overall_wo, avaerage_dice_train, global_step


def validate_vae(model, model_inferer, dataloader, val_dataset_len, device, crop_size, batch_size, epoch):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    # model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    # model_inferer = partial(sliding_window_inference, roi_size=crop_size, sw_batch_size=batch_size, predictor=model, overlap=0.5)
    val_loss = 0  # Initialize total loss accumulator
    quantization_losses = 0
    class_losses_sum_overall_wo = {'BG': 0, 'NC': 0, 'ED': 0, 'ET':0}
    class_losses_sum_overall = {'BG': 0, 'TC': 0, 'WT': 0, 'ET':0}
    # class_losses_sum_overall = {'BG': 0, 'NC': 0}
    hd_95npes = []
    # scale_factor = 1.0 + 0.1  # e.g. ±5%
    # shift_factor = 0.1
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)  # fixed seed
    batch0 = next(iter(dataloader))

    tmp = torch.stack(
        [batch0['t1n'], batch0['t2w'], batch0['t1c'], batch0['t2f']],
        dim=1
    ).to(device)
    
    C = tmp.shape[1]
    spatial_shape = tmp.shape[2:]
    
    # 🔴 generator is used here
    shared_noise = torch.randn(
        (1, C, *spatial_shape),
        generator=gen,
        device=device
    )
    
    # optional but correct scaling
    # shared_noise = shared_noise / shared_noise.std() * tmp.std()
    # signal_ratio = 0.5
    # noise_ratio = 0.5


    with torch.no_grad():  # Disable gradient computation for validation
        
        for batch in tqdm(dataloader, desc=f"Valditaion Network (Epoch {epoch})"):
            
                       
            images={}
            images_crop = {}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = batch[key]
                    #print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            # print("image shape with stacked modality is", images.shape)

            for key in ["t1n_crop", "t2w_crop", "t1c_crop", "t2f_crop"]:
                if key in batch:
                    images_crop[key] = batch[key]
                    #print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            images_crop = torch.stack([images_crop['t1n_crop'], images_crop['t2w_crop'], images_crop['t1c_crop'], images_crop['t2f_crop']], dim=1)
            
            # Get the segmentation mask from batch_data
            if 'mask' in batch:
                mask = batch['mask']

            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            mask_crop = batch['mask_crop']
                # print("image shape with seg_mask is", mask.shape)
            images = images.to(device)
            images_crop = images_crop.to(device)
            
            mask = mask.to(device)
            mask_crop = mask_crop.to(device)
            
            with torch.amp.autocast(device_type='cuda', enabled=False):  # Mixed precision context for validation
                
                # images = images * scale_factor + shift_factor
            #     noise = shared_noise.expand(
            #     images.shape[0], -1, *images.shape[2:]
            # )
        
            #     images = signal_ratio * images + noise_ratio * noise
                reconstruction, patch_predictions, patch_coordinates, patch_features = sliding_window_custom_collect(
    inputs=images,
    predictor=model,  # your model
    roi_size=(160, 160, 144),
    sw_batch_size=4,
    overlap=0.5,
    blend_mode="constant",
    device=device
)

                
                # reconstruction = model_inferer(images)
                # print("(type(reconstruction)", type(reconstruction))
                # print(reconstruction.shape)
                reconstruction_crop, quantized_loss, latent, feature = model(images_crop, mode='train', perform_retrieval=True)
                # print("(type(reconstruction_crop)", type(reconstruction_crop))
                # print(reconstruction_crop.shape)
                if reconstruction.shape[4] > 155:
                    # print("256.shape", reconstruction.shape)
                    reconstruction = reconstruction[:, :, :, :, :-1]

                ce_dice_loss = loss_function(reconstruction, mask)

                # print("ce_dice_loss", ce_dice_loss)

                
                combined_loss = dice_loss(reconstruction, mask)
                # print("combined_loss shape is", combined_loss.shape)
                combined_loss = combined_loss.mean(dim=0)
    
                # print(f"BG_loss_{combined_loss[0]}_____________NC_loss_{combined_loss[1]}_____________ED_loss_{combined_loss[2]}_____________ET_loss_{combined_loss[3]}")

                loss_BG = combined_loss[0]
                # class_losses_sum_overall+=
                # print("combined_loss shape is", combined_loss.shape)
                # print("combined_loss is", combined_loss)
                loss_NC = combined_loss[1]
                # print("loss_NC is", loss_NC.shape)
                # print("loss_NC is", loss_NC)
                # print("loss_NC is", loss_NC.item())
                loss_ED = combined_loss[2]
                # print("loss_ED is", loss_ED.shape)
                # print("loss_ED is", loss_ED)
                # print("loss_ED is", loss_ED.item())
                loss_EN = combined_loss[3]
                
    
                # quantization_loss = quantization_loss
                re_norm_combined_loss = ((loss_BG+loss_EN+loss_ED+loss_NC))
                # print("re_norm_combined_loss", re_norm_combined_loss)
                    
                # print("combined_loss", combined_loss)
                # # quantization_loss = quantization_loss/max_total_loss
                # # print("quantization_losses is", quantization_loss)
                batch_images = batch['mask'].shape[0]

                mask = torch.argmax(mask, dim=1)

                # mask_crop = torch.argmax(mask_crop, dim=1)

                
                # batch_unique_counts = []
            
                # for i in range(mask.shape[0]):
                #     unique_labels = torch.unique(mask_crop[i])
                #     unique_set = set(unique_labels.tolist())
                
                #     if unique_set == {0, 1, 2, 3}:
                #         batch_unique_counts.append(0)  # All labels present
                #     elif 1 not in unique_set:
                #         batch_unique_counts.append(1)  # Label 1 (e.g. TC) missing
                #     elif 3 not in unique_set:
                #         batch_unique_counts.append(2)  # Label 3 (ET) missing
                #     else:
                #         batch_unique_counts.append(0)  # An
                
                # # Optional: Convert to tensor
                # batch_unique_counts = torch.tensor(batch_unique_counts, device=device)
                
                # print("Unique label counts per sample:", batch_unique_counts)
                # router_loss = torch.nn.functional.cross_entropy(router_logits, batch_unique_counts)
                # print("router_loss", router_loss)

                
                mask = [(mask == 0), (mask == 1) | (mask == 3), (mask == 1) | (mask == 3) | (mask == 2), (mask == 3)]
                mask = torch.stack(mask, dim=1).float()

                # print("Updated mask shape is", mask.shape)  # Should be (8, 4, 120, 120, 96)

                # mask = torch.stack(mask, dim=1).float()
                # print("mask shape is", mask.shape)
                # reconstruction = torch.softmax(reconstruction, 1)
                reconstruction = torch.argmax(reconstruction, dim=1)
                reconstruction = [(reconstruction == 0), (reconstruction == 1) | (reconstruction == 3), (reconstruction == 1) | (reconstruction == 3) | (reconstruction == 2), (reconstruction == 3)]
                reconstruction = torch.stack(reconstruction, dim=1).float()
                # print("reconstruction shape is", reconstruction.shape)
                combined_loss_bts = dice_loss(reconstruction, mask)
                combined_loss_bts = combined_loss_bts.mean(dim=0)
    
                print(f"BG_loss_{combined_loss_bts[0]}__________TC_loss_{combined_loss_bts[1]}___________WT_loss_{combined_loss_bts[2]}_____________ET_loss_{combined_loss_bts[3]}")

                for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                    class_losses_sum_overall_wo[key]+=((combined_loss[idx].item())*batch_images)

                for idx, (key, value) in enumerate(class_losses_sum_overall.items()):
                    class_losses_sum_overall[key]+=((combined_loss_bts[idx].item())*batch_images)
                # print("class_losses_sum_overall", class_losses_sum_overall)
                
                # # # print("batch_images", batch_images)
                # print("class_losses_sum_overall", class_losses_sum_overall)
                
                # combined_loss = l1_loss(reconstruction.float(), mask.float())
                loss = (ce_dice_loss+quantized_loss)
                # print("total loss is", loss / 2)
                loss_val = loss*batch_images

                
    

    
            
            val_loss += loss_val.item()  # Accumulate the loss value
            # quantization_losses += quantization_loss

    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / val_dataset_len

    average_dice = 0
    for key, value in class_losses_sum_overall.items():
        class_losses_sum_overall[key] = value / val_dataset_len
        if key != 'BG':
            average_dice += value / val_dataset_len

    average_dice_val = average_dice/3
    # print("average_dice_val", average_dice_val)
    # hd_95es =torch.mean(torch.tensor(hd_95npes, dtype=torch.float32))
    # print("val_steps", hd_95es)
    # print("sum of hd95_es", sum(hd_95npes))
    # print("len of hd95_es", len(hd_95npes))
    # Return the average loss over the validation dataset
    return val_loss / val_dataset_len, class_losses_sum_overall, class_losses_sum_overall_wo, average_dice_val









def validate_vae_brats_val(model, model_inferer, dataloader, val_dataset_len, device, crop_size, batch_size, epoch):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    # model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    # model_inferer = partial(sliding_window_inference, roi_size=crop_size, sw_batch_size=batch_size, predictor=model, overlap=0.5)
    val_loss = 0  # Initialize total loss accumulator
    quantization_losses = 0
    class_losses_sum_overall_wo = {'BG': 0, 'NC': 0, 'ED': 0, 'ET':0}
    class_losses_sum_overall = {'BG': 0, 'TC': 0, 'WT': 0, 'ET':0}
    # class_losses_sum_overall = {'BG': 0, 'NC': 0}
    hd_95npes = []
    with torch.no_grad():  # Disable gradient computation for validation
        
        for batch in tqdm(dataloader, desc=f"Valditaion Network (Epoch {epoch})"):
            
                       
            images={}
            images_crop = {}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = batch[key]
                    #print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            case_id = batch.get("case_id")
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            # print("image shape with stacked modality is", images.shape)

            for key in ["t1n_crop", "t2w_crop", "t1c_crop", "t2f_crop"]:
                if key in batch:
                    images_crop[key] = batch[key]
                    #print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            images_crop = torch.stack([images_crop['t1n_crop'], images_crop['t2w_crop'], images_crop['t1c_crop'], images_crop['t2f_crop']], dim=1)
            
            
            images = images.to(device)
            

            
            with torch.amp.autocast(device_type='cuda', enabled=False):  # Mixed precision context for validation
                

                reconstruction, patch_predictions, patch_coordinates, patch_features = sliding_window_custom_collect(
    inputs=images,
    predictor=model,  # your model
    roi_size=(160, 160, 144),
    sw_batch_size=4,
    overlap=0.4,
    blend_mode="constant",
    device=device
)

                
                
                reconstruction = torch.argmax(reconstruction, dim=1)

                mask_path = batch.get("mask_path") or batch.get("mask_meta_dict", [{}])[0].get("filename_or_obj")
                
                if isinstance(mask_path, list):
                    print("mask_path_bef", mask_path)
                    mask_path = mask_path[0]
                    print("mask_path", mask_path)
                if not mask_path:
                    raise RuntimeError("Could not find path to mask to extract affine")
                affine = nib.load(mask_path).affine

                print("Affine matrix:\n", affine)
                print("Origin:", affine @ [0, 0, 0, 1])  # Gives world coordinate of first voxel

                # header = nib.load(mask_path).header.copy()


                # affine = nib.load(mask_path).affine

                pred_mask_np = reconstruction.squeeze(0).cpu().numpy().astype(np.uint8)
                print("pred_mask_np", pred_mask_np.shape)

                # ✅ Set save paths using case ID
                case_name = case_id[0] if isinstance(case_id, list) else str(case_id)
                save_dir = "analysis_outputs/brats_2023_fold0_test"
                os.makedirs(save_dir, exist_ok=True)
                pred_path = os.path.join(save_dir, f"{case_name}.nii.gz")

                brats_affine = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -239],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
                print("Corrected Affine matrix:\n", brats_affine)
                print("Origin:", brats_affine @ [0, 0, 0, 1])  # Gives world coordinate of first voxel
                # ✅ Save masks as NIfTI
                nib.save(nib.Nifti1Image(pred_mask_np, brats_affine), pred_path)
                # nib.save(nib.Nifti1Image(pred_mask_np, img_nii.affine, header), pred_path)
                
                
    
            
    return reconstruction








def analyze_fp_regions(final_segmentation_mask, final_GT_segmentation_mask):
    FP_mask = (final_segmentation_mask != final_GT_segmentation_mask) & (final_segmentation_mask != 0)
    FP_voxel_count = FP_mask.sum().item()

    edge_kernel = torch.ones((1, 1, 3, 3, 3), device=final_GT_segmentation_mask.device)
    edge_mask = F.conv3d(final_GT_segmentation_mask.unsqueeze(1).float(), edge_kernel, padding=1) > 0
    FP_near_boundary = (FP_mask & edge_mask.squeeze(1)).sum().item()
    FP_boundary_ratio = FP_near_boundary / (FP_voxel_count + 1e-8)

    FP_in_ET = (FP_mask & (final_GT_segmentation_mask == 3)).sum().item()
    FP_in_ED = (FP_mask & (final_GT_segmentation_mask == 2)).sum().item()
    FP_in_NC = (FP_mask & (final_GT_segmentation_mask == 1)).sum().item()
    FP_in_BG = (FP_mask & (final_GT_segmentation_mask == 0)).sum().item()

    return {
        "FP_Count": FP_voxel_count,
        "FP_Boundary_Ratio": FP_boundary_ratio,
        "FP_in_BG": FP_in_BG,
        "FP_in_NC": FP_in_NC,
        "FP_in_ED": FP_in_ED,
        "FP_in_ET": FP_in_ET
    }





def save_latent_with_labels(latent, gt_mask, case_id, size, save_dir="analysis_outputs/latent_vis_16D_hard_only_dec_enc_features"):
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)
    
    # Reshape latent and ground truth to 2D with labels
    gt_down = F.interpolate(gt_mask.float(), size=size, mode='nearest').squeeze(0).squeeze(0).long()
    gt_down = gt_mask
    latent = latent.squeeze(0).permute(1, 2, 3, 0)  # → (16, 16, 10, 256)
    latent_flat = latent.reshape(-1, 16).cpu().numpy()
    labels_flat = gt_down.reshape(-1).cpu().numpy()
    print("len of label flats", len(labels_flat))
    print("len of label flats", len(latent_flat))

    df = pd.DataFrame(latent_flat)
    df['label'] = labels_flat
    print(case_id)
    df['case_id'] = case_id[0] if isinstance(case_id, list) else case_id
    df.to_csv(os.path.join(save_dir, f"latent_{case_id}.csv"), index=False)

    print(f"✅ Latent CSV saved for case {case_id}")




def finalize_test_vae_analysis(metrics_list, save_dir="analysis_outputs"):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(save_dir, "per_case_metrics_16D_up_router_RAG.csv")
    df.to_csv(csv_path, index=False)

    wandb.log({"Per-Case Metrics": wandb.Table(dataframe=df)})

    worst_cases = df.sort_values("ET_Dice").head(5)
    print("\n🔍 Worst ET Dice Cases:")
    print(worst_cases[["case_id", "ET_Dice", "ET_HD95"]])

    # Dice Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(df["ET_Dice"], kde=True, bins=20, color="red", label="ET Dice")
    sns.histplot(df["TC_Dice"], kde=True, bins=20, color="orange", label="TC Dice")
    sns.histplot(df["WT_Dice"], kde=True, bins=20, color="blue", label="WT Dice")
    plt.title("Dice Score Distribution per Class")
    plt.xlabel("Dice Score")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    dice_plot_path = os.path.join(save_dir, "dice_distribution.png")
    plt.savefig(dice_plot_path)
    plt.close()
    wandb.log({"Dice Distribution": wandb.Image(dice_plot_path)})

    # Dice vs HD95 Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df["ET_Dice"], df["ET_HD95"], color="red", alpha=0.7, label="ET")
    plt.scatter(df["TC_Dice"], df["TC_HD95"], color="orange", alpha=0.7, label="TC")
    plt.scatter(df["WT_Dice"], df["WT_HD95"], color="blue", alpha=0.7, label="WT")
    plt.xlabel("Dice Score")
    plt.ylabel("HD95 (mm)")
    plt.title("Dice vs HD95")
    plt.axvline(0.75, color='gray', linestyle='--', label="Dice < 0.75")
    plt.axhline(5, color='gray', linestyle='--', label="HD95 > 5mm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, "dice_vs_hd95.png")
    plt.savefig(scatter_path)
    plt.close()
    wandb.log({"Dice vs HD95": wandb.Image(scatter_path)})

    # Dice vs Volume Plot
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.scatter(df["NC_Volume"], df["NC_Dice_wo"], color="purple", alpha=0.7)
    plt.title("NC Volume vs NC Dice")
    plt.xlabel("NC Tumor Volume (Voxels)")
    plt.ylabel("NC Dice")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.scatter(df["ED_Volume"], df["ED_Dice_wo"], color="green", alpha=0.7)
    plt.title("ED Volume vs ED Dice")
    plt.xlabel("ED Tumor Volume (Voxels)")
    plt.ylabel("ED Dice")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.scatter(df["ET_Volume"], df["ET_Dice_wo"], color="red", alpha=0.7)
    plt.title("ET Volume vs ET Dice")
    plt.xlabel("ET Tumor Volume (Voxels)")
    plt.ylabel("ET Dice")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.scatter(df["BG_Volume"], df["BG_Dice_wo"], color="blue", alpha=0.7)
    plt.title("BG Volume vs BG Dice")
    plt.xlabel("BG Volume (Voxels)")
    plt.ylabel("BG Dice")
    plt.grid(True)

    plt.tight_layout()
    dice_volume_plot_path = os.path.join(save_dir, "dice_vs_volume.png")
    plt.savefig(dice_volume_plot_path)
    plt.close()
    wandb.log({"Dice vs Volume (All Classes)": wandb.Image(dice_volume_plot_path)})

    return df, worst_cases

def test_vae(model_WT, model_inferer, dataloader, val_dataset_len, device):

    # import wandb
    # from torchcam.methods import GradCAM

    
    print("Validation in Progress")
    model_WT.eval()
    metrics_list = []
    count = 0
    class_losses_sum_overall = {'BG': 0, 'TC': 0, 'WT': 0, 'ET': 0}
    hd_95_app = {"WT_HD_95": 0, "TC_HD_95": 0, "ET_HD_95": 0}

    # ✅ Initialize Global Confusion Matrix
    global_confusion = np.zeros((4, 4), dtype=int)

    def save_average():
        if count > 0:
            avg_hd_95 = {key: hd_95_app[key] / count for key in hd_95_app}
            avg_dice = {key: class_losses_sum_overall[key] / count for key in class_losses_sum_overall}
            print("Final Averaged HD95:", avg_hd_95)
            print("Final Averaged Dice:", avg_dice)
            print("count is", count)
            return avg_hd_95, avg_dice
        else:
            print("No valid updates to compute average")
            return None, None

    def collect_voxel_features_for_csv(latent, label_mask, case_id, max_voxels_per_class=100, valid_classes=[0, 1, 2, 3]):
        B, C, D, H, W = latent.shape
        latent = latent.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        labels = label_mask.view(-1)
        for cls in valid_classes:
            idx = torch.where(labels == cls)[0]
            if idx.numel() == 0:
                continue
            if idx.numel() > max_voxels_per_class:
                idx = idx[torch.randperm(idx.shape[0])[:max_voxels_per_class]]
            selected_feats = latent[idx].cpu().numpy()
            selected_labels = np.full(idx.shape[0], cls)
            print("len of selected_labels", len(selected_labels))
            selected_case = [case_id[0] if isinstance(case_id, list) else case_id] * idx.shape[0]
            for feat_vec, label, case in zip(selected_feats, selected_labels, selected_case):
                voxel_feature_records.append(feat_vec.tolist() + [int(label), case])

    def fg_mean(tensor, mod, fg_mask):
        # print("tensor.squeeze().detach().cpu().numpy()", tensor.squeeze().detach().cpu().numpy().shape)
        data = tensor.squeeze().detach().cpu().numpy()
        # print("(fg_mask==3).sum()>0", data[fg_mask==3].mean())
        if mod == 't1c' and (fg_mask==3).sum()>0:
            print("(fg_mask==3).sum()>0", data[fg_mask==3].mean())
            return data[fg_mask==3].mean()
        elif mod == 't2f' and (fg_mask==2).sum()>0:
            return data[fg_mask==2].mean()
        elif mod == 't1n':
            return data[fg_mask>0].mean()
        elif mod == 't2w':
            return data[fg_mask>0].mean()
        else:
            return 0
                # mask = batch['mask_crop'].to(device)
                

    
   
    with torch.no_grad():
        for val_step, batch in enumerate(dataloader, start=1):
            images = {key: batch[key] for key in ["t1n", "t2w", "t1c", "t2f"]}
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1).to(device)

            t1n = batch['t1n_un'].to(device)
            t2w = batch['t2w_un'].to(device)
            t1c = batch['t1c_un'].to(device)
            t2f = batch['t2f_un'].to(device)

            print("images shpe is", images.shape)
            t1n_shape = images.shape

            images_crop = {key: batch[key] for key in ["t1n_crop", "t2w_crop", "t1c_crop", "t2f_crop"]}
            images_crop = torch.stack([images_crop['t1n_crop'], images_crop['t2w_crop'], images_crop['t1c_crop'], images_crop['t2f_crop']], dim=1).to(device)
            mask_crop = batch['mask_crop'].to(device)
            case_id = batch.get("case_id")
            
            mri_t1 = batch['t1n_normalized']
            mri_t2 = batch['t2w_normalized']
            mri_t1c = batch['t1c_normalized']
            mri_t2f = batch['t2f_normalized']
            mask = batch['mask'].to(device)
            fg_mask = (torch.argmax(mask, dim=1)).squeeze().detach().cpu().numpy()
            print("fg_mask", fg_mask.shape)
            fg_volume = (fg_mask>0).sum()

            gen = torch.Generator(device=device)
            gen.manual_seed(12345)  # fixed seed
            batch0 = next(iter(dataloader))
        
            tmp = torch.stack(
                [batch0['t1n'], batch0['t2w'], batch0['t1c'], batch0['t2f']],
                dim=1
            ).to(device)
            
            C = tmp.shape[1]
            spatial_shape = tmp.shape[2:]
            
            # 🔴 generator is used here
            shared_noise = torch.randn(
                (1, C, *spatial_shape),
                generator=gen,
                device=device
            )
            
            # optional but correct scaling
            shared_noise = shared_noise / shared_noise.std() * tmp.std()
            signal_ratio = 0.5
            noise_ratio = 0.5

            with torch.amp.autocast(device_type='cuda', enabled=False):
                noise = shared_noise.expand(
                images.shape[0], -1, *images.shape[2:]
            )
        
                images = signal_ratio * images + noise_ratio * noise


                reconstruction, patch_predictions, patch_coordinates, patch_features = sliding_window_custom_collect(
    inputs=images,
    predictor=model_WT,  # your model
    roi_size=(160, 160, 144),
    sw_batch_size=4,
    overlap=0.4,
    blend_mode="constant",
    device=device
)

                print("reconstruction shpe is", reconstruction.shape)

                reconstruction_crop, quantized_loss, latent, feature = model_WT(images_crop, mode='train', perform_retrieval=True)

                


                
                # reconstruction = model_inferer(images)
                print("reconstrion from monai is underway")
                combined_loss = dice_loss(reconstruction, mask).mean(dim=0)
                dice_BG_wo, dice_NC_wo, dice_ED_wo, dice_ET_wo = combined_loss

                mask = torch.argmax(mask, dim=1)

               
                
                final_GT_segmentation_mask = mask
                GT_unique_labels = (torch.unique(final_GT_segmentation_mask)).numel()
                print("GT_unique_labels", GT_unique_labels)

                gt_mask_input = (torch.argmax(mask_crop, dim=1)).unsqueeze(1)  # Shape: (B, 1, H, W, D)
                # print("gt_mask_input shape is", gt_mask_input.shape)
                # size = latent.shape[2:]
                # print("size is", size)
                # save_latent_with_labels(latent, gt_mask_input, case_id, size)


                # collect_voxel_features_for_csv(latent, final_GT_segmentation_mask, case_id)
                
                BG_voxel_count = (final_GT_segmentation_mask == 0).sum().item()
                NC_voxel_count = (final_GT_segmentation_mask == 1).sum().item()
                ED_voxel_count = (final_GT_segmentation_mask == 2).sum().item()
                ET_voxel_count = (final_GT_segmentation_mask == 3).sum().item()

                mask = [(mask == 0), (mask == 1) | (mask == 3), (mask == 1) | (mask == 3) | (mask == 2), (mask == 3)]
                mask = torch.stack(mask, dim=1).float()

                reconstruction = torch.argmax(reconstruction, dim=1)
                recons_unique_labels = (torch.unique(reconstruction)).numel()
                print("recons_unique_labels", recons_unique_labels)
                
                final_segmentation_mask = reconstruction
                reconstruction = [(reconstruction == 0), (reconstruction == 1) | (reconstruction == 3),
                                  (reconstruction == 1) | (reconstruction == 3) | (reconstruction == 2),
                                  (reconstruction == 3)]
                reconstruction = torch.stack(reconstruction, dim=1).float()
                combined_loss_bts = dice_loss(reconstruction, mask).mean(dim=0)
                dice_BG, dice_TC, dice_WT, dice_ET = combined_loss_bts

                hd_wt = compute_hausdorff_distance(reconstruction[:, 2:3], mask[:, 2:3], percentile=95)
                hd_tc = compute_hausdorff_distance(reconstruction[:, 1:2], mask[:, 1:2], percentile=95)
                hd_et = compute_hausdorff_distance(reconstruction[:, 3:4], mask[:, 3:4], percentile=95)

                hd_95 = {"WT_HD_95": hd_wt.item(), "TC_HD_95": hd_tc.item(), "ET_HD_95": hd_et.item()}
                dice_score = {"WT_Dice": dice_WT.item(), "TC_Dice": dice_TC.item(), "ET_Dice": dice_ET.item()}
                print(f"hd_95_____{hd_95}_______Dice_score______{dice_score}")

                if np.all(np.isfinite(np.array(list(hd_95.values())))):
                    hd_95_app = {key: hd_95_app[key] + hd_95[key] for key in hd_95_app}
                    class_losses_sum_overall = {key: (class_losses_sum_overall[key]) + (1 - v.item())for key, v in zip(class_losses_sum_overall.keys(), [dice_BG, dice_TC, dice_WT, dice_ET])}
                    count += 1

                ### 📦 Calculate Confusion Matrix and Accumulate
                y_true = final_GT_segmentation_mask.flatten().cpu().numpy()
                y_pred = final_segmentation_mask.flatten().cpu().numpy()

                conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
                global_confusion += conf_matrix  # ✅ Global accumulation




                # Use the meta dict from mask or any modality — they all contain affine
                mask_path = batch.get("mask_path") or batch.get("mask_meta_dict", [{}])[0].get("filename_or_obj")
                
                if isinstance(mask_path, list):
                    mask_path = mask_path[0]
                    print("mask_path", mask_path)
                if not mask_path:
                    raise RuntimeError("Could not find path to mask to extract affine")
                affine = nib.load(mask_path).affine

                # ✅ Convert predicted and GT masks to NumPy
                pred_mask_np = final_segmentation_mask.squeeze(0).cpu().numpy().astype(np.uint8)
                gt_mask_np = final_GT_segmentation_mask.squeeze(0).cpu().numpy().astype(np.uint8)
                
                # ✅ Set save paths using case ID
                case_name = case_id[0] if isinstance(case_id, list) else str(case_id)
                save_dir = "analysis_outputs/masks_train"
                os.makedirs(save_dir, exist_ok=True)
                pred_path = os.path.join(save_dir, f"pred_mask_{case_name}.nii.gz")
                gt_path = os.path.join(save_dir, f"gt_mask_{case_name}.nii.gz")
                
                # ✅ Save masks as NIfTI
                nib.save(nib.Nifti1Image(pred_mask_np, affine), pred_path)
                nib.save(nib.Nifti1Image(gt_mask_np, affine), gt_path)

                def safe_hd(val, default=50):
                    # Convert to Python float if it's a tensor
                    v = val.item() if hasattr(val, "item") else val
                    return v if math.isfinite(v) else default

                
                precision = precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
                recall = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
                f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
                fp_metrics = analyze_fp_regions(final_segmentation_mask, final_GT_segmentation_mask)
                metrics_list.append({
                    "case_id": case_id[0] if isinstance(case_id, list) else case_id,
                    "MRI_Shape": t1n_shape,
                    "WT_Dice": 1 - dice_WT.item(),
                    "TC_Dice": 1 - dice_TC.item(),
                    "ET_Dice": 1 - dice_ET.item(),
                    "WT_HD95": safe_hd(hd_wt),
                    "TC_HD95": safe_hd(hd_tc),
                    "ET_HD95": safe_hd(hd_et),
                    "fg_volume": int(fg_volume),
                    "fg_mean_t1n": fg_mean(t1n, 't1n', fg_mask),
                    "fg_mean_t1c": fg_mean(t1c, 't1c', fg_mask),
                    "fg_mean_t2w": fg_mean(t2w, 't2w', fg_mask),
                    "fg_mean_t2f": fg_mean(t2f, 't2f', fg_mask),
                    "BG_Dice_wo": 1 - dice_BG_wo.item(),
                    "NC_Dice_wo": 1 - dice_NC_wo.item(),
                    "ED_Dice_wo": 1 - dice_ED_wo.item(),
                    "ET_Dice_wo": 1 - dice_ET_wo.item(),
                    "NC_Volume": NC_voxel_count,
                    "ED_Volume": ED_voxel_count,
                    "ET_Volume": ET_voxel_count,
                    "BG_Volume": BG_voxel_count,
                    "BG_Precision": precision[0], "NC_Precision": precision[1], "ED_Precision": precision[2], "ET_Precision": precision[3],
                    "BG_Recall": recall[0], "NC_Recall": recall[1], "ED_Recall": recall[2], "ET_Recall": recall[3],
                    "BG_F1": f1[0], "NC_F1": f1[1], "ED_F1": f1[2], "ET_F1": f1[3],
                    
                    # ✅ TP/FP/FN
                    "BG_TP": conf_matrix[0, 0], "BG_FP": conf_matrix[:, 0].sum() - conf_matrix[0, 0], "BG_FN": conf_matrix[0, :].sum() - conf_matrix[0, 0],
                    "NC_TP": conf_matrix[1, 1], "NC_FP": conf_matrix[:, 1].sum() - conf_matrix[1, 1], "NC_FN": conf_matrix[1, :].sum() - conf_matrix[1, 1],
                    "ED_TP": conf_matrix[2, 2], "ED_FP": conf_matrix[:, 2].sum() - conf_matrix[2, 2], "ED_FN": conf_matrix[2, :].sum() - conf_matrix[2, 2],
                    "ET_TP": conf_matrix[3, 3], "ET_FP": conf_matrix[:, 3].sum() - conf_matrix[3, 3], "ET_FN": conf_matrix[3, :].sum() - conf_matrix[3, 3],
                    
                    # ✅ 🔥 Full confusion breakdown
                    "conf_BG_BG": conf_matrix[0, 0], "conf_BG_NC": conf_matrix[0, 1], "conf_BG_ED": conf_matrix[0, 2], "conf_BG_ET": conf_matrix[0, 3],
                    "conf_NC_BG": conf_matrix[1, 0], "conf_NC_NC": conf_matrix[1, 1], "conf_NC_ED": conf_matrix[1, 2], "conf_NC_ET": conf_matrix[1, 3],
                    "conf_ED_BG": conf_matrix[2, 0], "conf_ED_NC": conf_matrix[2, 1], "conf_ED_ED": conf_matrix[2, 2], "conf_ED_ET": conf_matrix[2, 3],
                    "conf_ET_BG": conf_matrix[3, 0], "conf_ET_NC": conf_matrix[3, 1], "conf_ET_ED": conf_matrix[3, 2], "conf_ET_ET": conf_matrix[3, 3],
                    "GT_Unique_Labels": GT_unique_labels,
                    "recons_unique_labels": recons_unique_labels,
                    # "Router_GT_Label": (batch_unique_counts[0]),          # Ground truth
                    # "Router_Pred_Label": (pred_unique_label[0].item()),    # Predicted from router logits
                    **fp_metrics
                })

                # # ✅ Finalize results
                hd_95, avg_dice = save_average()
            
                # --- Save as DataFrame
                df = pd.DataFrame(metrics_list)
                os.makedirs("analysis_outputs", exist_ok=True)
                df.to_csv("analysis_outputs/proposed_wo_vq_wt_noise_50%.csv", index=False)
            
                # ✅ Save the Global Confusion Matrix
                # class_names = ['BG', 'NC', 'ED', 'ET']
                # plt.figure(figsize=(8, 6))
                # sns.heatmap(global_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                # plt.xlabel('Predicted Class')
                # plt.ylabel('Ground Truth Class')
                # plt.title('Global Confusion Matrix')
                # plt.tight_layout()
                # plt.savefig("analysis_outputs/global_confusion_matrix.png")
                # plt.show()
                # cam_extractor = GradCAM(model_WT, target_layer='encoder.encoder1')  # Adjust this layer

                # # 2. Ensure gradients can be computed
                # model_WT.eval()
                # for p in model_WT.parameters():
                #     p.requires_grad = True
    
                # with torch.enable_grad():
                #     # cam_extractor = GradCAM(model_WT, target_layer='encoder.encoder1')  # Adjust layer name if needed
                #     input_img = images_crop[0].unsqueeze(0).requires_grad_()
                #     output = model_WT(input_img)  # forward pass
                #     # Choose a class index for which GradCAM is needed
                #     class_idx = 3
                #     cam = cam_extractor(input_tensor=input_img, class_idx=class_idx)
                #     heatmap = cam[0].cpu().numpy()
                #     slice_idx = heatmap.shape[-1] // 2
                #     mri_slice = batch['t1c'][0, :, :, slice_idx].cpu().numpy()
    
                #     fig, ax = plt.subplots()
                #     ax.imshow(mri_slice, cmap='gray')
                #     ax.imshow(heatmap[:, :, slice_idx], cmap='jet', alpha=0.4)
                #     ax.axis('off')
                #     ax.set_title(f"Case {case_id[0]} - GradCAM ET")
    
                #     # Log to wandb
                #     wandb.log({
                #         f"GradCAM_ET/case_{case_id[0]}": wandb.Image(fig)
                #     })
    
                #     plt.close(fig)
            
            
                yield final_GT_segmentation_mask, final_segmentation_mask, mri_t1, mri_t2, mri_t1c, mri_t2f, dice_score, hd_95, latent, case_id[0]



