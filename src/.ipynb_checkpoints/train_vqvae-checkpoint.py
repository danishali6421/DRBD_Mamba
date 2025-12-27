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
from src import VQVAE_mod
from src.VQVAE_mod import VQVAE
from monai.utils import first, set_determinism
from torch.optim import Adam
import torchvision
import pytorch_lightning as pl
import nibabel as nib

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from losses import DiceCELoss
# from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance, compute_average_surface_distance
from collections.abc import Callable, Sequence

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after


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
        reduction: LossReduction | str = LossReduction.MEAN,
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
                # print("target shape is", target.shape)
                # print("input shape is", input.shape)

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
# from losses import DiceCELoss
# from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance, compute_average_surface_distance


loss_function = DiceLoss(to_onehot_y=True, softmax=False)




# loss_function = DiceCELoss(to_onehot_y=True, softmax=False)

def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return optimizer

def training_step(self, batch, batch_idx):
    images, labels = batch['image'], batch['label']
        


scaler = GradScaler()

def train_vae(model, train_loader, train_dataset_len, optimizer, device):
    """
    Train the VAE model for one epoch with mixed precision.
    """
    model = model.to(device)
    #for epoch in range(n_epochs):
    model.train()
    scaler = GradScaler()
    epoch_loss = 0
    batch_count = 0
    for step, batch in enumerate(train_loader):      


        print("Training in Progress")
        # mask = batch['mask'].to(device)
        images={}
        for key in ["t1n", "t2w", "t1c", "t2f"]:
            if key in batch:
                images[key] = batch[key]
                #print(f"image shape with modality {key} is", batch[key].shape)
            else:
                raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
    
        # Stack modalities along the channel dimension (dim=1)
        images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
        # print("image shape with stacked modality is", images.shape)
        
        # Get the segmentation mask from batch_data
        if 'mask' in batch:
            mask = batch['mask']
            # print("image shape with seg_mask is", mask.shape)
        else:
            raise KeyError("Key 'segmentation' not found in batch_data") 
        
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device)
        mask = mask.to(device)
        # print("mask shape is", mask.shape)
    
        with autocast(device_type='cuda', enabled=False):

            output, latents, embloss = model(mask)
            print("Q loss is", embloss)
            # print("mask shape is", mask.shape)
            # print("output shape is", output.shape)
            loss = loss_function(output, mask)
            print("Dice loss is", loss)
            loss=loss+embloss
            # loss_mean = loss.mean()
            # print('train Loss: %.3f' % (loss.item()), 'codebook mean distance: %.6f' % (info[2].item()), 'codebook mean variance: %.6f' % (info[3].item()))
            # tensorboard_logs = {"train_loss": loss.item(), "mean_cb_distance": info[2].item(), "mean_cb_variance": info[3].item(),}
            
    
            
            # reconstruction, quantization_loss = model(images)

            # print("sum of reconstruction", torch.sum(output))
            # print("sum of mask", torch.sum(mask))
           
            print("total loss is", loss)
            batch_images = batch['mask'].shape[0]
            loss_tr = loss*batch_images
    
        scaler.scale(loss).backward()  # Scale loss and perform backward pass
        scaler.step(optimizer)  # Update model parameters
        scaler.update()
    
    
        epoch_loss += loss_tr.item()
    # Return the average loss over the epoch
    return epoch_loss / train_dataset_len






def validate_vae(model, model_inferer, dataloader, val_dataset_len, device):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    # model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0  # Initialize total loss accumulator
    with torch.no_grad():  # Disable gradient computation for validation
        
        for val_step, batch in enumerate(dataloader, start=1):
            
                       
            images={}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = batch[key]
                   # print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            # print("image shape with stacked modality is", images.shape)
            
            # Get the segmentation mask from batch_data
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            images = images.to(device)
            mask = mask.to(device)
            with autocast(device_type='cuda', enabled=False):  # Mixed precision context for validation
               
                output, latents, embloss = model(mask)
                print("Q loss is", embloss)
                loss = loss_function(output, mask)
                print("Dice loss is", loss)
                loss=loss+embloss
                # loss_mean = loss.mean()
                # print('train Loss: %.3f' % (loss.item()), 'codebook mean distance: %.6f' % (info[2].item()), 'codebook mean variance: %.6f' % (info[3].item()))
                # tensorboard_logs = {"train_loss": loss.item(), "mean_cb_distance": info[2].item(), "mean_cb_variance": info[3].item(),}
                
        
                
                # reconstruction, quantization_loss = model(images)
    
                # print("sum of reconstruction", torch.sum(output))
                # print("sum of mask", torch.sum(mask))
               
                print("total loss is", loss)
                batch_images = batch['mask'].shape[0]
                loss_val= loss*batch_images
        
    

    
            
            val_loss += loss_val.item()  # Accumulate the loss value

    # Return the average loss over the validation dataset
    return val_loss / val_dataset_len



def test_vae(model, dataloader, device, test_dataset_len, output_dir):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")
    print(device)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    test_loss = 0  # Initialize total loss accumulator
    class_losses_sum_overall = {'TC': 0, 'WT': 0, 'ET': 0}
    
    with torch.no_grad():  # Disable gradient computation for validation
        for test_step, batch in enumerate(dataloader, start=1):
            
                       
            images={}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = batch[key]
                   # print(f"image shape with modality {key} is", batch[key].shape)
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            # print("image shape with stacked modality is", images.shape)
            
            # Get the segmentation mask from batch_data
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            print(torch.cuda.is_available())

            images = images.to(device)
            mask = mask.to(device)
            print(torch.cuda.is_available())
            with autocast(device_type='cuda', enabled=False):  # Mixed precision context for validation
                # reconstruction_all, reconstruction_loss, quantization_loss = test_custom_sliding_window_inference(mask, model, (120, 120, 80), 4, 0.5) 
                output, latents, embloss = model(mask)
                loss = loss_function(output, mask)
                # loss_mean = loss.mean()
                # print('train Loss: %.3f' % (loss.item()), 'codebook mean distance: %.6f' % (info[2].item()), 'codebook mean variance: %.6f' % (info[3].item()))
                # tensorboard_logs = {"train_loss": loss.item(), "mean_cb_distance": info[2].item(), "mean_cb_variance": info[3].item(),}
                
        
                
                # reconstruction, quantization_loss = model(images)
    
                print("sum of reconstruction", torch.sum(output))
                print("sum of mask", torch.sum(mask))
               
                print("total loss is", loss)
                batch_images = batch['mask'].shape[0]
                loss_test= loss*batch_images


                reconstruction_all = output.squeeze().cpu().numpy()
                print("reconstruction_al", reconstruction_all.shape)
                reconstruction_all = nib.Nifti1Image(reconstruction_all, affine=np.eye(4))
                nib.save(reconstruction_all, os.path.join(output_dir, f'reconstructed_all_image_{test_step}.nii.gz'))

                mask = mask.squeeze().cpu().numpy()
                print("mask", mask.shape)
                mask = nib.Nifti1Image(mask, affine=np.eye(4))
                nib.save(mask, os.path.join(output_dir, f'mask_all_image_{test_step}.nii.gz'))
    

    
            
            test_loss += loss_test  # Accumulate the loss value
            
    # for key, value in class_losses_sum_overall.items():
    #     class_losses_sum_overall[key] = value / val_dataset_len

    # Return the average loss over the validation dataset
    return test_loss / test_dataset_len



















