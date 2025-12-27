import os
from torch.utils.data import Dataset, DataLoader
from src import dataset
from src.dataset import Dataloading
from config import configp
from config.configp import get_args  # Corrected import statement
from src.transformations import get_train_transforms, get_val_transforms
import monai
import torch
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor

#from monai.transforms import MapTransform, TransformBackends
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)
from monai.losses import FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss
import torch.nn as nn
import logging
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from src import train
from thop import profile
import os
import shutil
import tempfile
import time
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from sklearn.manifold import TSNE


####LDM

import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm


#from generative.networks.schedulers import DDPMScheduler
from monai.utils import first, set_determinism

from monai.losses import FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss
import torch.nn.functional as F
# from src.custom_gen_vqvae import VQVAE
import visdom
from src.train import train_vae, validate_vae, test_vae
# , validate_secondary, train_secondary, validate_vae_brats_val  # Import the train and validate functions
from monai.losses import DiceLoss

# Initialize Dice loss
# Initialize Visdom
# viz = visdom.Visdom()
import wandb
# import cuml
# from cuml.manifold import TSNE
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib  # Import matplotlib first
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import yaml
from monai.transforms import NormalizeIntensityd
from skimage.transform import resize
from functools import partial
from src.train import sliding_window_inference
from vqvae_unet import VQVAE_seq_unet
from sklearn.decomposition import PCA
# from torch.profiler import profile, record_function, ProfilerActivity
wandb.init(
    project="BTS_VAE_Model"
    
)





def run_pipeline(args):
    # Create datasets with splits
   # Create dataset instances for each split
    dice_loss = DiceLoss(to_onehot_y=False, softmax=False)
    #l1_loss = DiceLoss
    data_path=args.data_path
    file_path = args.file_path
    crop_size=args.crop_size
    modalities=args.modalities
    

    # train_dataset, val_dataset, test_dataset = Dataloading(file_path, crop_size, modalities, "training", "validation", "testing")
    train_dataset, val_dataset = Dataloading(file_path, crop_size, modalities, 0)

    # Print out dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    print("device is", device)




    
    if args.VQVAE:

        
        model = VQVAE_seq_unet(in_channels=4, out_channels=4, dropout_prob=0.0)

        
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-4)

        # Initialize the learning rate scheduler with adjusted parameters
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=500,  # Total epochs before LR reaches min
        eta_min=1e-6  # Minimum LR
    )

    
        
        
        model=model.to(device)
        print(model)
        
        if torch.cuda.is_available():
            input = torch.randn(1, 4, 128, 128, 128).to(device)
            flops, params = profile(model, (input,))
            print('Params = ' + str(params/1000**2) + 'M')
            print('FLOPs = ' + str(flops/1000**3) + 'G')


       
    
        # Create directories for saving model checkpoints
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)  # Create the directory if it does not existmodel,
       



        # Four MRi Visulazied
        def visualize_all_modalities_with_overlay(
            mri_t1, mri_t2, mri_t1c, mri_t2f,
            gt_mask, pred_mask,
            class_losses_sum_overall_wo, hd_95,
            title='MRI Modalities with GT and Predicted Overlay on T2Flair',
            wandb_log=True
        ):
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import ListedColormap
            from skimage.transform import resize
        
            # Squeeze and convert to numpy
            mri_t1 = mri_t1.squeeze().permute(2, 1, 0).cpu().numpy()
            mri_t2 = mri_t2.squeeze().permute(2, 1, 0).cpu().numpy()
            mri_t1c = mri_t1c.squeeze().permute(2, 1, 0).cpu().numpy()
            mri_t2f = mri_t2f.squeeze().permute(2, 1, 0).cpu().numpy()
            gt_mask = gt_mask.squeeze().permute(2, 1, 0).cpu().numpy()
            pred_mask = pred_mask.squeeze().permute(2, 1, 0).cpu().numpy()
        
            cmap = ListedColormap(['black', 'red', 'green', 'yellow'])
        
            slice_idx = mri_t1.shape[0] // 2  # Middle slice
        
            def normalize(img):
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                return np.stack([img]*3, axis=-1)
        
            def apply_colormap(base, mask):
                overlay = np.copy(base)
                overlay[mask == 1] = cmap(1)[:3]  # Red
                overlay[mask == 2] = cmap(2)[:3]  # Green
                overlay[mask == 3] = cmap(3)[:3]  # Yellow
                return overlay
        
            fig, axs = plt.subplots(1, 6, figsize=(24, 5))
            plt.subplots_adjust(wspace=0.01)
        
            # MRI modalities
            axs[0].imshow(normalize(mri_t1[slice_idx]))
            axs[0].set_title("T1")
            axs[1].imshow(normalize(mri_t2[slice_idx]))
            axs[1].set_title("T2")
            axs[2].imshow(normalize(mri_t1c[slice_idx]))
            axs[2].set_title("T1c")
            axs[3].imshow(normalize(mri_t2f[slice_idx]))
            axs[3].set_title("T2Flair")
        
            # Overlays on T2Flair
            base_t2f = normalize(mri_t2f[slice_idx])
            gt_overlay = apply_colormap(base_t2f, gt_mask[slice_idx])
            pred_overlay = apply_colormap(base_t2f, pred_mask[slice_idx])
        
            axs[4].imshow(gt_overlay)
            axs[4].set_title("GT on T2Flair")
            axs[5].imshow(pred_overlay)
            axs[5].set_title("Pred on T2Flair")
        
            for ax in axs:
                ax.axis("off")
        
            # Save
            plot_path = f"{title.replace(' ', '_')}_vis.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
        
            if wandb_log:
                import wandb
                wandb.log({f"{title}_visualization": wandb.Image(plot_path)})
        
            print(f"Saved visualization to: {plot_path}")





        def visualize_worst_slice_error(
        mri_t1, mri_t2, mri_t1c, mri_t2f,
        gt_mask, pred_mask, class_losses_sum_overall_wo,
        title='Worst Error Slice - GT vs Pred Overlay',
        wandb_log=True
    ):
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import matplotlib.gridspec as gridspec
            import numpy as np
            import wandb
        
            # Convert tensors to numpy
            mri_t1 = mri_t1.squeeze().permute(2, 0, 1).cpu().numpy()
            mri_t2 = mri_t2.squeeze().permute(2, 0, 1).cpu().numpy()
            mri_t1c = mri_t1c.squeeze().permute(2, 0, 1).cpu().numpy()
            mri_t2f = mri_t2f.squeeze().permute(2, 0, 1).cpu().numpy()
            gt_mask = gt_mask.squeeze().permute(2, 0, 1).cpu().numpy()
            pred_mask = pred_mask.squeeze().permute(2, 0, 1).cpu().numpy()
        
            def normalize(img):
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return np.stack([img] * 3, axis=-1)
        
            def apply_colormap(base, mask):
                overlay = np.copy(base)
                overlay[mask == 1] = [1, 0, 0]       # NCR/ECT - red
                overlay[mask == 2] = [0, 0.5, 0]       # Edema - yellow
                overlay[mask == 3] = [1, 1, 0]     # GD-Enhancing Tumor - green
                return overlay
        
            # Get worst slice
            error_map = (gt_mask != pred_mask).astype(int)
            error_per_slice = error_map.sum(axis=(1, 2))
            worst_slice_idx = np.argmax(error_per_slice)
        
            base = normalize(mri_t2f[worst_slice_idx])
            gt_overlay = apply_colormap(base, gt_mask[worst_slice_idx])
            modality_imgs = {
                "T1": normalize(mri_t1[worst_slice_idx]),
                "T1ce": normalize(mri_t1c[worst_slice_idx]),
                "T2": normalize(mri_t2[worst_slice_idx]),
                "FLAIR": normalize(mri_t2f[worst_slice_idx])
            }
        
            # Create 3-row x 4-col grid (GT spans 2x2 area at top-right)
            fig = plt.figure(figsize=(14, 8))  # ← default white background
            gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.25], hspace=0.04, wspace=0.04)
        
            # Plot 2×2 MRI modalities
            positions = [("T1", 0, 0), ("T1ce", 0, 1), ("T2", 1, 0), ("FLAIR", 1, 1)]
            for label, r, c in positions:
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(modality_imgs[label])
                ax.axis("off")
                ax.text(0.03, 0.97, label, transform=ax.transAxes,
                        ha='left', va='top', fontsize=12, fontweight='bold',
                        color='white', bbox=dict(facecolor='black', alpha=0.7, edgecolor='black'))
        
            # GT Overlay (spans 2x2)
            ax_gt = fig.add_subplot(gs[0:2, 2:4])
            ax_gt.imshow(gt_overlay)
            ax_gt.axis("off")
            ax_gt.text(0.03, 0.97, "Ground Truth", transform=ax_gt.transAxes,
                       ha='left', va='top', fontsize=12, fontweight='bold',
                       color='white', bbox=dict(facecolor='black', alpha=0.7, edgecolor='black'))
        
            # Legend (bottom row)
            ax_legend = fig.add_subplot(gs[2, :])
            ax_legend.axis('off')
        
            region_colors = {
                "Background": "black",
                "Peritumoral Edema": "#006400",
                "Enhancing Tumor": "yellow",
                "Necrotic Core": "red"
            }
        
            box_width = 1 / len(region_colors)
            for i, (label, color) in enumerate(region_colors.items()):
                rect = patches.Rectangle((i * box_width, 0.2), box_width * 0.95, 0.6,
                                         linewidth=1.5, edgecolor=color, facecolor=color,
                                         transform=ax_legend.transAxes)
                ax_legend.add_patch(rect)
                if i==0: 
                # Always use black text (since background is white)
                    ax_legend.text((i + 0.475) * box_width, 0.5, label,
                                   transform=ax_legend.transAxes, ha='center', va='center',
                                   fontsize=10, fontweight='bold', color='white')
                else:
                    ax_legend.text((i + 0.475) * box_width, 0.5, label,
                                   transform=ax_legend.transAxes, ha='center', va='center',
                                   fontsize=10, fontweight='bold', color='black')
        
            # Save and log
            plot_path = f"{title.replace(' ', '_')}_slice_{worst_slice_idx}_vis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
            if wandb_log:
                wandb.log({f"{title}_worst_slice_vis": wandb.Image(plot_path)})
        
            print(f"[✓] Saved worst error slice visualization to: {plot_path}")




        
        # Determine whether to resume training or start from scratch
        if args.vqvae_training:
            
            start_epoch = 0  # Default start epoch
            if args.resume:
                print("Resume training from epoch")
                checkpoint_path = os.path.join(checkpoint_dir, 'systematic_fold_1_sfc_vq_rem.pth')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)  # Load the latest checkpoint
                    # print("checkpoint.keys()", checkpoint['model_state_dict'].keys())
                    model.load_state_dict(checkpoint['model_state_dict'], strict= False)  # Restore model state
                    
                    start_epoch = checkpoint['epoch']  # Restore the last epoch
                    print("checkpoint['epoch']", checkpoint['epoch'] )

                    
            else:
                print("No checkpoint found. Starting training from scratch.")
                checkpoint_path = os.path.join(checkpoint_dir, 'systematic_fold_1_sfc_vq_rem.pth')
                checkpoint_path_100_epoch = os.path.join(checkpoint_dir, 'systematic_fold_1_sfc_vq_rem_100_epoch.pth')
                # if os.path.exists(checkpoint_path):
                #     os.remove(checkpoint_path)  # Remove existing checkpoint if starting from scratch
        
            # Training and validation loop
            total_start = time.time()
            num_epochs = 2000  # Set the number of epochs
            # model_inferer = partial(
            #     sliding_window_inference,
            #     roi_size=crop_size,
            #     sw_batch_size=4,
            #     predictor=model,
            #     overlap=0.5,
            # )
            
            train_loss_lis=[]
            val_loss_lis=[]

            
            def plot_encodings_sum(encodings_sumation, epoch):
                """Plot all 1024 encoding sums as a bar chart on the same figure and log to WandB."""
                # Move the tensor to the CPU, check its shape, and convert to numpy
                encodings_sumation_cpu = encodings_sumation.cpu().numpy()
            
                # Debugging: Print the shape to ensure it has 1024 values
                print(f"Shape of encodings_sumation at epoch {epoch}: {encodings_sumation_cpu.shape}")
            
                # If the tensor is not 1D, flatten it
                if len(encodings_sumation_cpu.shape) > 1:
                    encodings_sumation_cpu = encodings_sumation_cpu.flatten()
                
                # Plot all 1024 values as a bar chart
                plt.figure(figsize=(15, 6))  # Adjust figure size for better visibility
                plt.bar(range(len(encodings_sumation_cpu)), encodings_sumation_cpu)
                plt.xlabel("Encoding Index")
                plt.ylabel("Sum of Encodings")
                plt.title(f"Encodings Sum at Epoch {epoch}")
            
                # Log the figure to WandB as an image
                wandb.log({f'encodings_sum_epoch_{epoch}': wandb.Image(plt)})
                
                # Close the plot to free memory
                plt.close()

            

            def log_latent_space(latents, latent_name, epoch, n_clusters=4):
                """
                Visualizes and logs each latent space separately to W&B using GPU-based t-SNE and Seaborn scatter plot.
                
                Args:
                latents: The latent vectors extracted from a specific codebook, shape (batch_size, num_channels, height, width, depth)
                latent_name: The name to log for each quantized latent (e.g., z_quantized0)
                epoch: Current epoch to track the training phase
                n_clusters: Number of clusters for coloring the latent points
                """
                # Reshape latents to (batch_size * spatial dimensions, num_channels)
                latents_reshaped = latents.view(-1, latents.size(1))  # Shape (batch_size * 30 * 30 * 20, num_channels)
            
                # Detach from computation graph before converting to NumPy
                latents_reshaped = latents_reshaped.detach().cpu().numpy()
            
                # Check the number of samples in reshaped latents
                num_samples = latents_reshaped.shape[0]
            
                # Set the perplexity, ensure it is smaller than the number of samples
                perplexity = min(30, num_samples - 1)
            
                # Apply GPU-based t-SNE to reduce the high-dimensional latent space to 2D
                tsne = cuml.TSNE(n_components=2, perplexity=perplexity)
            
                try:
                    latents_2d = tsne.fit_transform(latents_reshaped)
            
                    # Apply KMeans to find clusters in the latent space
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels = kmeans.fit_predict(latents_reshaped)
            
                    # Create a DataFrame for easier plotting
                    df = pd.DataFrame(latents_2d, columns=['x', 'y'])
                    df['cluster'] = cluster_labels
            
                    # Create the scatter plot using Seaborn
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', style='cluster', markers='o', s=100)
                    plt.title(f"{latent_name} Latent Space at Epoch {epoch}")
                    plt.xlabel('TSNE Component 1')
                    plt.ylabel('TSNE Component 2')
            
                    # Save the plot to a temporary file
                    plot_path = f"{latent_name}_latent_space_epoch.png"
                    plt.savefig(plot_path)
                    plt.close()
            
                    # Log the image to W&B
                    wandb.log({f"{latent_name}_latent_space": wandb.Image(plot_path)})
            
                except ValueError as e:
                    print(f"Error visualizing latent space for {latent_name}: {e}")
        

            utilized_encoding = torch.zeros(512).to(device)
            best_average_dice_val = float('inf')
            global_steps = 0
            for epoch in range(start_epoch, num_epochs):  # Resume from the last saved epoch or start from scratch
                print("Training in progress")
                print(f"Epoch {epoch+1}/{num_epochs}")  # Print current epoch number
                
                train_loss, class_losses_sum_overall, class_losses_sum_overall_wo, avaerage_dice_train, global_step = train_vae(model, train_loader, len(train_dataset), optimizer, device, epoch, global_steps)
                global_steps += global_step
                wandb.log({'epoch': epoch + 1, 'train_loss': train_loss / 2})
                wandb.log({'epoch': epoch + 1, 'class_losses_sum_train_overall': class_losses_sum_overall})
                wandb.log({'epoch': epoch + 1, 'class_losses_sum_train_overall_wo': class_losses_sum_overall_wo})
               
                torch.cuda.empty_cache()
                # Validate the model every 5 epochs
                # print("faiss_built_this_epoch", faiss_built_this_epoch)
                if (epoch + 1) % (50 if epoch < 250 else 1) == 0:
                    print("Validation in progress")
                    model_inferer = partial(
            sliding_window_inference,
            roi_size=crop_size,
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
            mode="gaussian"
        )
                    # mask_validation, reconstruction_validation, mask_val, val_loss, class_losses_sum_overall_val, class_losses_sum_overall_val_wo, Q_loss_val, embedding0 = validate_vae(cond_model, model, model_inferer, val_loader, len(val_dataset), device)
                    val_loss, class_losses_sum_overall_val, class_losses_sum_overall_val_wo, average_dice_val = validate_vae(model, model_inferer, val_loader, len(val_dataset), device, crop_size, args.batch_size, epoch)
                    
                    wandb.log({'epoch': epoch + 1, 'val_loss': val_loss / 2})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_val_overall_val': class_losses_sum_overall_val})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_overall_val_wo': class_losses_sum_overall_val_wo})
                    
                    print(f'Epoch: {epoch}_Train_Loss: {train_loss}_Val_Loss: {val_loss}_class_losses_sum_overall:{class_losses_sum_overall}_class_losses_sum_overall_val:{class_losses_sum_overall_val}')  # Print validation loss
                    # print(f'class_losses_sum_train_overall: {class_losses_sum_overall}')
                    scheduler.step()
                   
                    torch.save({
                        'epoch': epoch + 1,  # Save current epoch
                        'model_state_dict': model.state_dict(),  # Save model state
                        'optimizer1_state_dict': optimizer.state_dict(),  # Save optimizer state
                        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                        # 'Utilized_encoding' : utilized_encoding,
                        # 'embedding0' : embedding0,
                        # 'optimizer2_state_dict': optimizer2.state_dict(),
                        'train_loss': train_loss,  # Save training loss
                        'val_loss': val_loss,  # Save validation loss
                        'train_loss_list': train_loss_lis,
                        'val_loss_lis': val_loss_lis
                    }, checkpoint_path)
                    if epoch+1 == 100:
                        torch.save({
                        'epoch': epoch + 1,  # Save current epoch
                        'model_state_dict': model.state_dict(),  # Save model state
                        'optimizer1_state_dict': optimizer.state_dict(),  # Save optimizer state
                        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                        # 'Utilized_encoding' : utilized_encoding,
                        # 'embedding0' : embedding0,
                        # 'optimizer2_state_dict': optimizer2.state_dict(),
                        'train_loss': train_loss,  # Save training loss
                        'val_loss': val_loss,  # Save validation loss
                        'train_loss_list': train_loss_lis,
                        'val_loss_lis': val_loss_lis
                    }, checkpoint_path_100_epoch)
                    print(f'Saved checkpoint to {checkpoint_path}')  # Confirm checkpoint saving
                    print("average_dice_val", average_dice_val)
                    print("best_average_dice_val", best_average_dice_val)
                    if average_dice_val <= best_average_dice_val:
                        best_average_dice_val = average_dice_val
                        best_model_path = checkpoint_path.replace(".pth", "_best.pth")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer1_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'average_dice_val': average_dice_val
                        }, best_model_path)
                        print(f"🔥 New best model saved at epoch {epoch+1} with Dice {average_dice_val:.4f} -> {best_model_path}")
                        wandb.log({'best_average_dice_val': best_average_dice_val})

                    # ✅ Save metadata to CSV log
                    import csv
                    from datetime import datetime
                
                    log_path = "best_model_log.csv"
                    log_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": epoch,
                        "average_dice_val": average_dice_val,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "class_losses_sum_val_overall_val": class_losses_sum_overall_val,
                        "checkpoint_path": checkpoint_path
                    }
                
                    file_exists = os.path.isfile(log_path)
                    with open(log_path, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=log_data.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(log_data)
                    torch.cuda.empty_cache()
                if (epoch + 1) % 2000 == 0:

                    test_loss, class_losses_sum_overall_test, class_losses_sum_overall_test_wo, avaerage_dice_test = validate_vae(model, model_inferer, val_loader, len(val_dataset), device, crop_size, args.batch_size, epoch)
                    
                  
                    wandb.log({'epoch': epoch + 1, 'test_loss': test_loss / 2})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_test_overall_test': class_losses_sum_overall_test})
                    wandb.log({'epoch': epoch + 1, 'class_losses_sum_overall_test_wo': class_losses_sum_overall_test_wo})
                    # wandb.log({'epoch': epoch + 1, 'Q_loss_test': Q_loss_test})
            total_time = time.time() - total_start
            print(f"train completed, total time: {total_time}.")
            
        else:
            model_inferer = partial(
            sliding_window_inference,
            roi_size=crop_size,
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
            mode="gaussian"
        )
            print("No Training argument provided")
            def log_latent_space(latents, latent_name, epoch, mapped_labels, counts, n_clusters=4):
                """
                Visualizes and logs each latent space separately to W&B using GPU-based t-SNE and Seaborn scatter plot.
                
                Args:
                latents: The latent vectors extracted from a specific codebook, shape (batch_size, num_channels, height, width, depth)
                latent_name: The name to log for each quantized latent (e.g., z_quantized0)
                epoch: Current epoch to track the training phase
                n_clusters: Number of clusters for coloring the latent points
                """
                # Reshape latents to (batch_size * spatial dimensions, num_channels)
                latents = latents.contiguous()
                latents_reshaped = latents.view(-1, latents.size(1))  # Shape (batch_size * 30 * 30 * 20, num_channels)
            
                # Detach from computation graph before converting to NumPy
                latents_reshaped = latents_reshaped.detach().cpu().numpy()
            
                # Check the number of samples in reshaped latents
                num_samples = latents_reshaped.shape[0]
                print("num_samples", num_samples)
                # Set the perplexity, ensure it is smaller than the number of samples
                perplexity = min(30, num_samples - 1)
            
                # Apply GPU-based t-SNE to reduce the high-dimensional latent space to 2D
                tsne = cuml.TSNE(n_components=2, perplexity=perplexity)
            
                try:
                    latents_2d = tsne.fit_transform(latents_reshaped)

                    # Apply KMeans to find clusters in the latent space
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels = kmeans.fit_predict(latents_reshaped)
                    
                    # Create a DataFrame for easier plotting
                    df = pd.DataFrame(latents_2d, columns=['x', 'y'])
                    df['cluster'] = cluster_labels
                    
                    # Count the number of points in each cluster
                    cluster_counts = df['cluster'].value_counts()
                    print("cluster_counts", cluster_counts)
                    
                    # Sort clusters by size (largest to smallest)
                    sorted_clusters = cluster_counts.index.tolist()
                    
                    # Map the sorted clusters to 'BG', 'ED', 'NC', 'ET'
                    if len(mapped_labels) == 4:
                        
                        label_mapping = {
                            sorted_clusters[0]: mapped_labels[0],  # Largest cluster
                            sorted_clusters[1]: mapped_labels[1],  # Second largest cluster
                            sorted_clusters[2]: mapped_labels[2],  # Third largest cluster
                            sorted_clusters[3]: mapped_labels[3]    # Smallest cluster
                        }
                    elif len(mapped_labels) ==3:
                        label_mapping = {
                            sorted_clusters[0]: mapped_labels[0],  # Largest cluster
                            sorted_clusters[1]: mapped_labels[1],  # Second largest cluster
                            sorted_clusters[2]: mapped_labels[2],  # Third largest cluster
                            sorted_clusters[3]: 'ET'    # Smallest cluster
                        }
                    else:
                        label_mapping = {
                            sorted_clusters[0]: mapped_labels[0],  # Largest cluster
                            sorted_clusters[1]: 'NC',  # Second largest cluster
                            sorted_clusters[2]: 'NC',  # Third largest cluster
                            sorted_clusters[3]: 'ET'    # Smallest cluster
                        }
                    # Apply the label mapping to the clusters
                    df['cluster'] = df['cluster'].map(label_mapping)
                    
                    # Define consistent colors and markers for each label
                    palette = {'BG': 'black', 'NC': 'red', 'ED': 'green', 'ET': 'blue'}
                    markers = ['o', 's', 'D', '^']  # Different marker styles for each cluster
                    
                    # Create the scatter plot using Seaborn with the updated labels and color palette
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=df, x='x', y='y', hue='cluster', style='cluster', 
                                    palette=palette, markers=markers, s=100)
                    
                    plt.title(f"{latent_name} Latent Space at Epoch {epoch}")
                    plt.xlabel('TSNE Component 1')
                    plt.ylabel('TSNE Component 2')
                    
                    # Calculate the centroid (mean x and y coordinates) for each cluster
                    centroids = df.groupby('cluster')[['x', 'y']].mean()
                    print("Centroids:", centroids.index.tolist())
                    print("Cluster Counts:", cluster_counts.index.tolist())
                    # Add annotations for data point counts at the centroid of each cluster
                    y_offset = 0.05  # Adjust this to increase/decrease spacing between counts
                    initial_y = 0.95 - y_offset  # Start just below total count
                    total_count = len(df)
                    # Loop through cluster indices and labels
                    i=0
                    for cluster_index, label in label_mapping.items():
                        print("cluster_index", cluster_index)
                        print("label", label)
                        
                        if cluster_index in cluster_counts.index:  # Check if label exists
                            count = cluster_counts[cluster_index]  # Get count for the label
                            centroid_x = centroids.loc[label, 'x']
                            centroid_y = centroids.loc[label, 'y']
                            print("percetnage count", counts[i])
                            print(f"Label: {label}, Count: {count}")  # Print found label and count
                            
                            
                    
                            # Annotate the cluster counts below the total count in a stacked manner
                            plt.text(0.95, initial_y, f'{label}: {count}', fontsize=12, weight='bold', 
                                     color=palette[label], ha='right', va='top', transform=plt.gca().transAxes, 
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                            
                            # Update the initial_y for the next count
                            initial_y -= y_offset  # Move down for the next count
                        else:
                            print(f"Label {label} not found in cluster_counts")  # Print missing label

                    
                    # Calculate the total count of all data points
                    total_count = len(df)
                    
                    # Annotate the total count of all data points
                    plt.text(0.95, 0.95, f'Total Count: {total_count}', fontsize=14, weight='bold', 
                             color='black', ha='right', va='top', transform=plt.gca().transAxes, 
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                    
                    # Save the plot to a temporary file
                    plot_path = f"{latent_name}_latent_space_epoch.png"
                    plt.savefig(plot_path)
                    plt.close()
                    
                    # Log the image to W&B
                    wandb.log({f"{latent_name}_latent_space": wandb.Image(plot_path)})



            
                except ValueError as e:
                    print(f"Error visualizing latent space for {latent_name}: {e}")
    
           
            checkpoint_path = os.path.join(checkpoint_dir, 'seg_mambastyle_test_slit_500_epochs training.pth')
            checkpoint = torch.load(checkpoint_path)
            print("checkpoint['epoch']", checkpoint['epoch'])


            
    
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(torch.cuda.is_available())
            print("device is", device)
            model.to(device) 
            # output_dir='./output'
    
        #     # test_loss, test_fm_loss, test_miss_loss = test_vae(model, test_loader, device, output_dir)
            num_classes = 4  # Adjust according to your dataset
            accumulated_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
       
            

           
            recons, class_losses_sum_overall, class_losses_sum_overall_wo, average_dice_val = validate_vae(model, model_inferer, val_loader, len(val_dataset), device, crop_size, args.batch_size, 1)
            print("class_losses_sum_overall", class_losses_sum_overall)



##########################Visualization####################################
            
#             for mask_validation, reconstruction_validation, mri_t1, mri_t2, mri_t1c, mri_t2f, class_losses_sum_overall_wo, hd_95, z, case_id in test_vae(model, model_inferer, val_loader, len(val_dataset), device):
                
#                 print("Testing Done")
#                 # label_mapping = {0: "WT", 1: "TC", 2: "ET"}
#                 label_mapping = {
# 0: "BG",
# 1: "NC",
# 2: "ED",
# 3: "ET"
# }
#                 for batch in range(mask_validation.shape[0]):
#                     mri_t1 = mri_t1.squeeze(0)
#                     mri_t1c = mri_t1c.squeeze(0)
#                     mri_t2 = mri_t2.squeeze(0)
#                     mri_t2f = mri_t2f.squeeze(0)
                    
                    
#                     visualize_all_modalities_with_overlay(mri_t1, mri_t2, mri_t1c, mri_t2f, mask_validation, reconstruction_validation, class_losses_sum_overall_wo, hd_95, title=case_id)

#                     visualize_worst_slice_error(mri_t1, mri_t2, mri_t1c, mri_t2f, mask_validation, reconstruction_validation, class_losses_sum_overall_wo, title=f'mask_validation_Data_{batch}')
                    
                    
#                     print("mask_validation.shape[0]", mask_validation.shape[0])  
#                     mask_validation_batch = mask_validation[batch]
                    
#                     reconstruction_validation_batch = reconstruction_validation[batch]
                    
                    
#                     total_count = mask_validation_batch.numel()
#                     print("total_count", total_count)
#                     label_counts = {label: (mask_validation_batch == label).sum().item() for label in np.unique(mask_validation_batch)}
#                     print("label_counts", label_counts)
                    
#                     # Sort labels and collect mappings
#                     sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
#                     mapped_labels = [label_mapping[label] for label, _ in sorted_labels]
#                     counts = [count / total_count for _, count in sorted_labels]







if __name__ == "__main__":
    args = get_args()  # Import arguments from the config file
    print("args:", args)  # Print arguments for verification
    run_pipeline(args)  # Run the data pipeline
