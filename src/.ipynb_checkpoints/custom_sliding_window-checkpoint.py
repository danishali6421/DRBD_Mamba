import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def compute_gaussian_importance_map(patch_size, sigma_scale=0.125):
    center_coords = [np.arange(s) - (s - 1) / 2 for s in patch_size]
    squared_distance = sum([(cc / (sigma_scale * (s - 1) / 2)) ** 2 for cc, s in zip(center_coords, patch_size)])
    gaussian_map = np.exp(-0.5 * squared_distance)
    gaussian_map = torch.tensor(gaussian_map, dtype=torch.float32)
    return gaussian_map / gaussian_map.max()

def sliding_window_custom_collect(
    inputs,
    predictor,
    roi_size,
    overlap=0.25,
    sw_batch_size=1,
    blend_mode="constant",   # ✅ Added blend mode
    sigma_scale=0.125,
    device=None
):
    device = device or inputs.device
    batch_size, channels, *spatial_dims = inputs.shape
    # channels = 3
    roi_size = list(roi_size)

    assert len(roi_size) == len(spatial_dims), "roi_size must match input spatial dims"

    stride = [int(r * (1 - overlap)) for r in roi_size]
    stride = [s if s > 0 else 1 for s in stride]

    scan_steps = [((s - r) // st + 1) for s, r, st in zip(spatial_dims, roi_size, stride)]
    patch_starts = []
    for i, (s, r, st) in enumerate(zip(spatial_dims, roi_size, stride)):
        stops = list(range(0, s - r + 1, st))
        if stops[-1] != s - r:
            stops.append(s - r)
        patch_starts.append(stops)

    patches = []
    patch_coords = []
    feature_vectors = []

    importance_map = torch.ones([1, 1, *roi_size], device=device)
    if blend_mode == "gaussian":
        importance_map = compute_gaussian_importance_map(roi_size, sigma_scale=sigma_scale).to(device)
        importance_map = importance_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    output_shape = (batch_size, channels, *spatial_dims)  # Assuming 4 channels output (segmentation)
    output = torch.zeros(output_shape, device=device)
    count_map = torch.zeros((batch_size, 1, *spatial_dims), device=device)

    # Start sliding
    total_patches = len(patch_starts[0]) * len(patch_starts[1]) * len(patch_starts[2])
    patch_idx = 0

    for x in patch_starts[0]:
        for y in patch_starts[1]:
            for z in patch_starts[2]:
                patch = inputs[..., x:x+roi_size[0], y:y+roi_size[1], z:z+roi_size[2]]
                pred, features = predictor(patch)  # pred [1, 4, H, W, D], features [1, 16, H, W, D]

                pred = pred.to(device)
                features = features.to(device)

                patches.append(pred.detach())
                feature_vectors.append(features.detach())
                patch_coords.append((x, y, z))

                # Blending
                output[..., x:x+roi_size[0], y:y+roi_size[1], z:z+roi_size[2]] += pred * importance_map
                count_map[..., x:x+roi_size[0], y:y+roi_size[1], z:z+roi_size[2]] += importance_map

                patch_idx += 1

    # Final normalization
    output = output / (count_map + 1e-5)

    return output, patches, patch_coords, feature_vectors



















# it was working now above i adedit willreturn un bleneded poacehs
    



# def sliding_window_custom_collect(
#     inputs: torch.Tensor,  # (1, C, H, W, D)
#     model: Callable,       # model expects (B, C, H, W, D)
#     roi_size: Tuple[int, int, int],
#     overlap: float = 0.5,
#     sw_batch_size: int = 1,
# ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[Tuple[slice, slice, slice]]]:
#     """
#     Customized Sliding Window Inference:
#     - Returns final output after blending.
#     - Also collects intermediate feature vectors and outputs for patches.
#     """
#     assert inputs.dim() == 5, "Input must be (B, C, H, W, D)"
#     device = inputs.device

#     batch_size, in_channels, H, W, D = inputs.shape
#     assert batch_size == 1, "Batch size must be 1 for now"

#     stride = [int(r * (1 - overlap)) for r in roi_size]
#     stride = [max(1, s) for s in stride]

#     # Compute grid
#     xs = list(range(0, max(H - roi_size[0] + 1, 1), stride[0]))
#     ys = list(range(0, max(W - roi_size[1] + 1, 1), stride[1]))
#     zs = list(range(0, max(D - roi_size[2] + 1, 1), stride[2]))

#     if xs[-1] + roi_size[0] < H:
#         xs.append(H - roi_size[0])
#     if ys[-1] + roi_size[1] < W:
#         ys.append(W - roi_size[1])
#     if zs[-1] + roi_size[2] < D:
#         zs.append(D - roi_size[2])

#     output_channels = 4  # assuming model outputs 4 channels (your setting)
#     feature_channels = 16  # assuming features are 16D vectors (your setting)

#     output_prob_sum = torch.zeros((1, output_channels, H, W, D), device=device)
#     feature_vec_sum = torch.zeros((1, feature_channels, H, W, D), device=device)
#     count_map = torch.zeros((1, 1, H, W, D), device=device)

#     feature_patch_list = []
#     output_patch_list = []
#     location_list = []

#     all_slices = []
#     for x in xs:
#         for y in ys:
#             for z in zs:
#                 x_slice = slice(x, x + roi_size[0])
#                 y_slice = slice(y, y + roi_size[1])
#                 z_slice = slice(z, z + roi_size[2])
#                 all_slices.append((x_slice, y_slice, z_slice))

#     for start_idx in range(0, len(all_slices), sw_batch_size):
#         batch_slices = all_slices[start_idx:start_idx+sw_batch_size]

#         patch_list = []
#         for (x_slice, y_slice, z_slice) in batch_slices:
#             patch = inputs[:, :, x_slice, y_slice, z_slice]
#             patch_list.append(patch)

#         patches = torch.cat(patch_list, dim=0)  # (sw_batch_size, C, roi_size...)
#         recon_batch, feature_batch = model(patches)  # should return 2 outputs

#         for idx, (x_slice, y_slice, z_slice) in enumerate(batch_slices):
#             reconstruction_patch = recon_batch[idx:idx+1]
#             feature_patch = feature_batch[idx:idx+1]

#             output_prob_sum[:, :, x_slice, y_slice, z_slice] += reconstruction_patch
#             feature_vec_sum[:, :, x_slice, y_slice, z_slice] += feature_patch
#             count_map[:, :, x_slice, y_slice, z_slice] += 1

#             feature_patch_list.append(feature_patch.detach().cpu())
#             output_patch_list.append(reconstruction_patch.detach().cpu())
#             location_list.append((x_slice, y_slice, z_slice))

#     output_prob_final = output_prob_sum / (count_map + 1e-5)
#     feature_vec_final = feature_vec_sum / (count_map + 1e-5)

#     return output_prob_final, feature_vec_final, feature_patch_list, output_patch_list, location_list