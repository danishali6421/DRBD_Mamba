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
from torch.nn.modules.loss import _Loss
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
from collections.abc import Callable, Sequence
# Initialize Visdom
# viz = visdom.Visdom(
#     'http://localhost'  # URL of the Visdom server
#     # port=8097,                  # Port where the Visdom server is running
#     # env='my_experiment',        # Environment for the session                 # Connection timeout in seconds
# )


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

dice_loss = DiceLoss(to_onehot_y=False, softmax=False)


def train_ldm(autoencoder, cond_model, ldm_model, train_loader, train_dataset, optimizer, device, inferer, tar_batch_size):
    """
    Train the VAE model for one epoch with mixed precision.
    """    
        
        
       # with autocast(enabled=False):
        
    # ldm_model = ldm_model.to(device)
    #for epoch in range(n_epochs):
    ldm_model.train()
    autoencoder.eval()
    scaler = GradScaler()
    epoch_loss = 0
    for step, batch in enumerate(train_loader):
        with torch.no_grad():   
            with autocast(device_type='cuda', enabled=True):

                images = {}
                for key in ["t1n", "t2w", "t1c", "t2f"]:
                    if key in batch:
                        images[key] = batch[key]
                    else:
                        raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
            
                # Stack modalities along the channel dimension (dim=1)
                images = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
                if 'mask' in batch:
                    mask = batch['mask']
                    # print("image shape with seg_mask is", mask.shape)
                else:
                    raise KeyError("Key 'segmentation' not found in batch_data") 
    
                # images_tensor = images_tensor.to(device)
                mask = mask.to(device)
                mask_up = mask[:,1:,:,:,:]
                print(torch.cuda.is_available())
            
                # # mask = mask.to(device)
                # print("device", device)
                # z_full = autoencoder.encode_stage_2_inputs(mask_up).to(device)
                # print("Latent full shape is", z_full.shape)
                images = images.to(device)
                print("images", images.shape)
                autoencoder_latent=autoencoder.encoder(mask_up) 
                autoencoder_latent = autoencoder.bottleneck(autoencoder_latent)
                autoencoder_latent=autoencoder.conv1(autoencoder_latent) 
                autoencoder_latent=autoencoder.conv2(autoencoder_latent) 
                autoencoder_latent_indices=autoencoder.quantizer0.quantize(autoencoder_latent)
                autoencoder_latent_indices_embeddingsss = autoencoder.quantizer0.embed(autoencoder_latent_indices)
                autoencoder_latent_indices = autoencoder_latent_indices.long()
                x_bot, x_bottt, quantized, quantized_loss = cond_model(images, autoencoder_latent)
                # autoencoder_latent_indices = autoencoder_latent_indices.view(autoencoder_latent_indices.shape[0], -1)
                # print("autoencoder_latent_indices", autoencoder_latent_indices)
                x_bot = torch.argmax(x_bot, dim=1)
                embeddingsss = autoencoder.quantizer0.embed(x_bot)
                z_quantized0_post = autoencoder.conv3(embeddingsss)
                z_quantized0_post = autoencoder.conv4(z_quantized0_post)
                reconstruction = autoencoder.decoder(z_quantized0_post)
                reconstruction = autoencoder.segmentation(reconstruction)
                reconstruction = torch.argmax(reconstruction, dim=1)
                reconstruction = (reconstruction==1).unsqueeze(1).float()
                verify_gt=torch.argmax(mask, dim=1)
                verify_gt = (verify_gt==1).unsqueeze(1).float()
                print("dic eloss isssss", dice_loss(reconstruction, verify_gt).mean(dim=0))
                print("reconstruction shape is", reconstruction.shape)
                reconstruction_latent=autoencoder.encoder(reconstruction) 
                reconstruction_latent = autoencoder.bottleneck(reconstruction_latent)
                reconstruction_latent=autoencoder.conv1(reconstruction_latent) 
                reconstruction_latent=autoencoder.conv2(reconstruction_latent) 
                reconstruction_latent_indices=autoencoder.quantizer0.quantize(reconstruction_latent)
                reconstruction_latent_indices_embeddingsss = autoencoder.quantizer0.embed(reconstruction_latent_indices)
                reconstruction_latent_indices = reconstruction_latent_indices.long()
                print("len where equal", torch.sum(reconstruction_latent_indices == autoencoder_latent_indices))
                # print("cosine loss with gt and predicted", cosine_loss(reconstruction_latent.view(autoencoder_latent_indices.shape[0], -1), autoencoder_latent.view(autoencoder_latent_indices.shape[0], -1)))

        # modality_keys = list(images.keys())
        # modality_to_remove = modality_keys[torch.randint(0, len(modality_keys), (1,)).item()]
        # # modality_to_remove = 't2'
        # images_missing = {key: (images[key] if key != modality_to_remove else torch.ones_like(images[key])) for key in images}
        # images_missing_tensor = torch.stack([images_missing['t1n'], images_missing['t2w'], images_missing['t1c'], images_missing['t2f']], dim=1)
        # print("images_missing_tensor", images_missing_tensor.shape)
        # # mean_dim0 = torch.mean(images_missing_tensor)
        # # mean_value = mean_dim0.item()
        # # print("mean with miss modality", mean_dim0) 
        # images_missing_tensor = images_missing_tensor.to(device)
        # images_missing_tensor = autoencoder.encode_stage_2_inputs(images_missing_tensor)

        
        # batch_size = images_missing_tensor.size(0)
    # if batch_size==tar_batch_size:
        
        print("Training in Progress")
        # mask = batch['mask'].to(device)
        # print("mask shape is", mask.shape)
        # print("z shape is", z.shape)
        # images={}
        # for key in ["t1", "t2", "t1ce", "flair"]:
        #     if key in batch:
        #         images[key] = batch[key]
        #         #print(f"image shape with modality {key} is", batch[key].shape)
        #     else:
        #         raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
    
        # # Stack modalities along the channel dimension (dim=1)
        # images = torch.stack([images['t1'], images['t2'], images['t1ce'], images['flair']], dim=1)
        # images=images.to(device)
        # print("image shape with stacked modality is", images.shape)
        # if z_full.shape[0] != mask_up.shape[0]:
        #     z_full = z_full[:mask_up.shape[0]]  # Slice z to match the batch size of mask
        #     print("Adjusted z_full shape is", z_full.shape)
        # noise = torch.randn_like(z_full).to(device)
        noise = autoencoder_latent

        optimizer.zero_grad(set_to_none=True)
        
        
     
        with autocast(device_type='cuda', enabled=True):
            # Generate random noise
            # Create timesteps
            # print("Time Setps Started")
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (mask.shape[0],), device=mask.device
            ).long()

            print("timesteps", train_dataset)
            # print("Time Setps Done")

            # Get model prediction
            noise_pred = inferer(
                inputs=reconstruction_latent, autoencoder_model=autoencoder, diffusion_model=ldm_model, noise=noise, timesteps=timesteps, condition=x_bottt)
            # print("Noise Started")

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        batch_images = batch['mask'].shape[0]
        epoch_loss += loss.item() * batch_images
    # else:
    #     print("Batch Size is", batch_size)
          
    # Return the average loss over the epoch
    return epoch_loss / train_dataset







def validate_ldm(autoencoder, cond_model, ldm_model, dataloader, val_dataset, device, inferer, tar_batch_size, scheduler):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")


    ldm_model = ldm_model.to(device)
    #for epoch in range(n_epochs):
    ldm_model.eval()
    autoencoder.eval()
    # autoencoder.eval()
    # scaler = GradScaler()
    val_loss = 0
    class_losses_sum_overall_wo = {"BG":0, 'NC': 0, 'ED': 0, 'ET': 0}
    class_losses_sum_overall = {"BG":0, 'TC': 0, 'WT': 0, 'ET': 0}
    with torch.no_grad(): 
        
        for step, batch in enumerate(dataloader):
            images = {}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = batch[key]
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            # images_tensor = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            # images_tensor = images_tensor.to(device)
            mask = mask.to(device)
            mask_up = mask[:,1:,:,:,:]
            print(torch.cuda.is_available())
        #batch = pad_batch(batch, tar_batch_size)          
            with autocast(device_type='cuda', enabled=True):

                
                
                print("device", device)
                z_full = autoencoder.encode_stage_2_inputs(mask_up).to(device)
                
                print("Latent full shape is", z_full.shape)

                modality_keys = list(images.keys())
                modality_to_remove = modality_keys[torch.randint(0, len(modality_keys), (1,)).item()]
                # modality_to_remove = 't2'
                images_missing = {key: (images[key] if key != modality_to_remove else torch.ones_like(images[key])) for key in images}
                images_missing_tensor = torch.stack([images_missing['t1n'], images_missing['t2w'], images_missing['t1c'], images_missing['t2f']], dim=1)
                print("images_missing_tensor", images_missing_tensor.shape)
                # mean_dim0 = torch.mean(images_missing_tensor)
                # mean_value = mean_dim0.item()
                # print("mean with miss modality", mean_dim0) 
                images_missing_tensor = images_missing_tensor.to(device)
                
                print("Validation in Progress")
               
                if z_full.shape[0] != images_missing_tensor.shape[0]:
                    z_full = z_full[:mask.shape[0]]  # Slice z to match the batch size of mask
                    print("Adjusted z_full shape is", z_full.shape)
                noise = torch.randn_like(z_full).to(device)
                print("noise device is", noise.device)
                # noise = torch.randn((1, 3, 24, 24, 16))
                # noise = noise.to(device)
                scheduler.set_timesteps(num_inference_steps=1)
                reconstruction = inferer.sample(
                    input_noise=noise, autoencoder_model=autoencoder, diffusion_model=ldm_model, scheduler=scheduler, conditioning=images_missing_tensor
                )
                print("reconstruction shape is", reconstruction.shape)
                # loss = F.mse_loss(noise_pred.float(), noise.float())
                combined_loss = dice_loss(reconstruction, mask)
                # print("combined_loss shape is", combined_loss.shape)
                combined_loss = combined_loss.mean(dim=0)
    
                print(f"BG_loss_{combined_loss[0]}__________NC_loss_{combined_loss[1]}___________ED_loss_{combined_loss[2]}_____________ET_loss_{combined_loss[3]}")
                # print(f"BG_loss_{combined_loss[0]}__________NC_loss_{combined_loss[1]}")
    
                loss_BG = combined_loss[0]
               
                loss_NC = combined_loss[1]
             
                loss_ED = combined_loss[2]
               
                loss_EN = combined_loss[3]
                
                
                
                re_norm_combined_loss = ((loss_BG+loss_NC+loss_ED+loss_EN))
                print("re_norm_combined_loss", re_norm_combined_loss)
    
                batch_images = batch['mask'].shape[0]
    
    
                for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                    class_losses_sum_overall_wo[key]+=((combined_loss[idx].item())*batch_images)
    
                
                mask = torch.argmax(mask, dim=1)
                print("mask shape is", mask.shape)
                mask = [(mask == 0), (mask == 1) | (mask == 3), (mask == 1) | (mask == 3) | (mask == 2), (mask == 3)]
                mask = torch.stack(mask, dim=1).float()
    
                print("Updated mask shape is", mask.shape)  # Should be (8, 4, 120, 120, 96)
    
                reconstruction = torch.argmax(reconstruction, dim=1)
                reconstruction = [(reconstruction == 0), (reconstruction == 1) | (reconstruction == 3), (reconstruction == 1) | (reconstruction == 3) | (reconstruction == 2), (reconstruction == 3)]
                reconstruction = torch.stack(reconstruction, dim=1).float()
                print("reconstruction shape is", reconstruction.shape)
                combined_loss_bts = dice_loss(reconstruction, mask)
                combined_loss_bts = combined_loss_bts.mean(dim=0)
    
                print(f"BG_loss_{combined_loss_bts[0]}__________TC_loss_{combined_loss_bts[1]}___________WT_loss_{combined_loss_bts[2]}_____________ET_loss_{combined_loss_bts[3]}")
                for idx, (key, value) in enumerate(class_losses_sum_overall.items()):
                    class_losses_sum_overall[key]+=((combined_loss_bts[idx].item())*batch_images)
        
            loss = (re_norm_combined_loss)
            print("total loss is", loss / 4)
            loss_val = loss*batch_images
            val_loss += loss_val.item()
        
    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / val_dataset
    
    for key, value in class_losses_sum_overall.items():
        class_losses_sum_overall[key] = value / val_dataset

                


    
    

    # Return the average loss over the validation dataset
    return val_loss / val_dataset, class_losses_sum_overall_wo, class_losses_sum_overall




def test_ldm(autoencoder, ldm_model, dataloader, val_dataset, device, inferer, tar_batch_size, scheduler):
    """
    Validate the VAE model on the validation dataset.
    """
    print("Validation in Progress")


    ldm_model = ldm_model.to(device)
    #for epoch in range(n_epochs):
    ldm_model.eval()
    autoencoder.eval()
    # autoencoder.eval()
    # scaler = GradScaler()
    z_min=0
    z_max=0
    val_loss = 0
    class_losses_sum_overall_wo = {"BG":0, 'NC': 0, 'ED': 0, 'ET': 0}
    class_losses_sum_overall = {"BG":0, 'TC': 0, 'WT': 0, 'ET': 0}
    with torch.no_grad(): 
        
        for step, batch in enumerate(dataloader):
            images = {}
            for key in ["t1n", "t2w", "t1c", "t2f"]:
                if key in batch:
                    images[key] = batch[key]
                else:
                    raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
        
            # Stack modalities along the channel dimension (dim=1)
            # images_tensor = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
            if 'mask' in batch:
                mask = batch['mask']
                # print("image shape with seg_mask is", mask.shape)
            else:
                raise KeyError("Key 'segmentation' not found in batch_data") 

            # images_tensor = images_tensor.to(device)
            mask = mask.to(device)
            mask_up = mask[:,1:,:,:,:]
            print(torch.cuda.is_available())
        #batch = pad_batch(batch, tar_batch_size)          
            with autocast(device_type='cuda', enabled=True):

                
                
                print("device", device)
                z_full = autoencoder.encode_stage_2_inputs(mask_up).to(device)
                
                print("Latent full shape is", z_full.shape)

                modality_keys = list(images.keys())
                modality_to_remove = modality_keys[torch.randint(0, len(modality_keys), (1,)).item()]
                # modality_to_remove = 't2'
                images_missing = {key: (images[key] if key != modality_to_remove else torch.ones_like(images[key])) for key in images}
                images_missing_tensor = torch.stack([images_missing['t1n'], images_missing['t2w'], images_missing['t1c'], images_missing['t2f']], dim=1)
                print("images_missing_tensor", images_missing_tensor.shape)
                # mean_dim0 = torch.mean(images_missing_tensor)
                # mean_value = mean_dim0.item()
                # print("mean with miss modality", mean_dim0) 
                images_missing_tensor = images_missing_tensor.to(device)
                
                print("Validation in Progress")
               
                if z_full.shape[0] != images_missing_tensor.shape[0]:
                    z_full = z_full[:mask.shape[0]]  # Slice z to match the batch size of mask
                    print("Adjusted z_full shape is", z_full.shape)
                noise = torch.randn_like(z_full).to(device)
                print("noise device is", noise.device)
                # noise = torch.randn((1, 3, 24, 24, 16))
                # noise = noise.to(device)
                scheduler.set_timesteps(num_inference_steps=5)
                reconstruction, latent = inferer.sample(
                    input_noise=noise, autoencoder_model=autoencoder, diffusion_model=ldm_model, scheduler=scheduler, conditioning=images_missing_tensor
                )
                print("reconstruction shape is", reconstruction.shape)
                print("z_full max value is", torch.mean(torch.max(z_full)))
                print("z_full min value is", torch.mean(torch.min(z_full)))
                z_min=min(z_min, (torch.mean(torch.min(z_full))))
                z_max=max(z_max, (torch.mean(torch.max(z_full))))
                print("latent max value is", torch.mean(torch.max(latent)))
                print("latent min value is", torch.mean(torch.min(latent)))
                denoise_loss = F.mse_loss(latent.float(), z_full.float())
                print("denoise loss is", denoise_loss)
                combined_loss = dice_loss(reconstruction, mask)
                # print("combined_loss shape is", combined_loss.shape)
                combined_loss = combined_loss.mean(dim=0)
    
                print(f"BG_loss_{combined_loss[0]}__________NC_loss_{combined_loss[1]}___________ED_loss_{combined_loss[2]}_____________ET_loss_{combined_loss[3]}")
                # print(f"BG_loss_{combined_loss[0]}__________NC_loss_{combined_loss[1]}")
    
                loss_BG = combined_loss[0]
               
                loss_NC = combined_loss[1]
             
                loss_ED = combined_loss[2]
               
                loss_EN = combined_loss[3]
                
                
                
                re_norm_combined_loss = ((loss_BG+loss_NC+loss_ED+loss_EN))
                print("re_norm_combined_loss", re_norm_combined_loss)
    
                batch_images = batch['mask'].shape[0]
    
    
                for idx, (key, value) in enumerate(class_losses_sum_overall_wo.items()):
                    class_losses_sum_overall_wo[key]+=((combined_loss[idx].item())*batch_images)
    
                
                mask = torch.argmax(mask, dim=1)
                print("mask shape is", mask.shape)
                mask = [(mask == 0), (mask == 1) | (mask == 3), (mask == 1) | (mask == 3) | (mask == 2), (mask == 3)]
                mask = torch.stack(mask, dim=1).float()
    
                print("Updated mask shape is", mask.shape)  # Should be (8, 4, 120, 120, 96)
    
                reconstruction = torch.argmax(reconstruction, dim=1)
                reconstruction = [(reconstruction == 0), (reconstruction == 1) | (reconstruction == 3), (reconstruction == 1) | (reconstruction == 3) | (reconstruction == 2), (reconstruction == 3)]
                reconstruction = torch.stack(reconstruction, dim=1).float()
                print("reconstruction shape is", reconstruction.shape)
                combined_loss_bts = dice_loss(reconstruction, mask)
                combined_loss_bts = combined_loss_bts.mean(dim=0)
    
                print(f"BG_loss_{combined_loss_bts[0]}__________TC_loss_{combined_loss_bts[1]}___________WT_loss_{combined_loss_bts[2]}_____________ET_loss_{combined_loss_bts[3]}")
                for idx, (key, value) in enumerate(class_losses_sum_overall.items()):
                    class_losses_sum_overall[key]+=((combined_loss_bts[idx].item())*batch_images)
        
            loss = (re_norm_combined_loss)
            print("total loss is", loss / 4)
            loss_val = loss*batch_images
            val_loss += loss_val.item()
        
    for key, value in class_losses_sum_overall_wo.items():
        class_losses_sum_overall_wo[key] = value / val_dataset
    print("class_losses_sum_overall_wo", class_losses_sum_overall_wo)
    
    for key, value in class_losses_sum_overall.items():
        class_losses_sum_overall[key] = value / val_dataset
    print("class_losses_sum_overall", class_losses_sum_overall)
    print("max od z_max is", (z_max))
    print("min od z_min is", (z_min))
                


    
    

    # Return the average loss over the validation dataset
    return val_loss / val_dataset, class_losses_sum_overall_wo, class_losses_sum_overall







# def test_ldm(autoencoder, ldm_model, dataloader, test_dataset, device, inferer, tar_batch_size):
#     """
#     Validate the VAE model on the validation dataset.
#     """
#     print("Validation in Progress")


#     ldm_model = ldm_model.to(device)
#     #for epoch in range(n_epochs):
#     ldm_model.eval()
#     autoencoder.eval()
#     # autoencoder.eval()
#     # scaler = GradScaler()
#     test_loss = 0
#     for step, batch in enumerate(dataloader):
#         #batch = pad_batch(batch, tar_batch_size)
        
#         with torch.no_grad():   
#             with autocast(device_type='cuda', enabled=True):

#                 images = {}
#                 for key in ["t1n", "t2w", "t1c", "t2f"]:
#                     if key in batch:
#                         images[key] = batch[key]
#                     else:
#                         raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
            
#                 # Stack modalities along the channel dimension (dim=1)
#                 images_tensor = torch.stack([images['t1n'], images['t2w'], images['t1c'], images['t2f']], dim=1)
#                 if 'mask' in batch:
#                     mask = batch['mask']
#                     # print("image shape with seg_mask is", mask.shape)
#                 else:
#                     raise KeyError("Key 'segmentation' not found in batch_data") 
    
#                 images_tensor = images_tensor.to(device)
#                 mask = mask.to(device)
#                 mask_up = mask[:,1:,:,:,:]
#                 print(torch.cuda.is_available())
                
#                 print("device", device)
#                 z_full = autoencoder.encode_stage_2_inputs(mask_up).to(device)
                
#                 print("Latent full shape is", z_full.shape)

#                 modality_keys = list(images.keys())
#                 modality_to_remove = modality_keys[torch.randint(0, len(modality_keys), (1,)).item()]
#                 # modality_to_remove = 't2'
#                 images_missing = {key: (images[key] if key != modality_to_remove else torch.ones_like(images[key])) for key in images}
#                 images_missing_tensor = torch.stack([images_missing['t1n'], images_missing['t2w'], images_missing['t1c'], images_missing['t2f']], dim=1)
#                 print("images_missing_tensor", images_missing_tensor.shape)
#                 # mean_dim0 = torch.mean(images_missing_tensor)
#                 # mean_value = mean_dim0.item()
#                 # print("mean with miss modality", mean_dim0) 
#                 images_missing_tensor = images_missing_tensor.to(device)
#                 # images_missing_tensor = autoencoder.encode_stage_2_inputs(images_missing_tensor).to(device)
        
                
#                 # batch_size = images_missing_tensor.size(0)
#             # if batch_size==tar_batch_size:
                
#                 print("Testing in Progress")
#                 # mask = batch['mask'].to(device)
#                 # print("mask shape is", mask.shape)
#                 # print("z shape is", z.shape)
#                 # images={}
#                 # for key in ["t1", "t2", "t1ce", "flair"]:
#                 #     if key in batch:
#                 #         images[key] = batch[key]
#                 #         #print(f"image shape with modality {key} is", batch[key].shape)
#                 #     else:
#                 #         raise KeyError(f"Key {key} not found in batch_data")  # Ensure key exists
            
#                 # # Stack modalities along the channel dimension (dim=1)
#                 # images = torch.stack([images['t1'], images['t2'], images['t1ce'], images['flair']], dim=1)
#                 # images=images.to(device)
#                 # print("image shape with stacked modality is", images.shape)
#                 if z_full.shape[0] != images_missing_tensor.shape[0]:
#                     z_full = z_full[:mask.shape[0]]  # Slice z to match the batch size of mask
#                     print("Adjusted z_full shape is", z_full.shape)
#                 noise = torch.randn_like(z_full).to(device)
        
    
#                 timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (mask.shape[0],), device=mask.device).long()

#                 print("timesteps", test_dataset)
#                 # print("Time Setps Done")
    
#                 # Get model prediction
#                 noise_pred = inferer(
#                     inputs=mask_up, autoencoder_model=autoencoder, diffusion_model=ldm_model, noise=noise, timesteps=timesteps, condition=None)
#                 # print("Noise Started")
    
#                 loss = F.mse_loss(noise_pred.float(), noise.float())

#                 test_loss += loss.item()


    
    

#     # Return the average loss over the validation dataset
#     return test_loss / test_dataset























