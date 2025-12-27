# import logging
# import os
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset, DataLoader
# from config.configp import get_args
# from src.transformations import get_train_transforms, get_val_transforms

# import cv2
# import numpy as np
# from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
# from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
# import numpy as np
# import random




# class BrainTumorDataset(Dataset):
#     def __init__(self, data_path, modalities, crop_size, split='train'):
#         self.data_path = data_path
#         self.modalities = modalities
#         self.crop_size = crop_size
#         self.split = split
#         # self.morph_ops = morph_ops
        
#         self.folders = sorted([os.path.join(data_path, folder) for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])
#         random.seed(42)  # Set seed for reproducibility
#         random.shuffle(self.folders)  # Shuffle the folders
#         # Split the dataset
#         train_size = int(0.70 * len(self.folders))
#         val_size = int(0.10 * len(self.folders))
#         test_size = len(self.folders) - train_size - val_size
#         print(self.modalities)
#         if self.split == 'train':
#             print("train_data")
#             self.folders = self.folders[0:train_size]
#             self.transforms = get_train_transforms(crop_size)
#         elif self.split == 'val':
#             self.folders = self.folders[train_size:train_size+val_size]
#             self.transforms = get_val_transforms(crop_size)
#         elif self.split == 'test':
#             self.folders = self.folders[train_size+val_size:train_size+val_size+test_size]
#             self.transforms = get_val_transforms(crop_size)  # Same as validation
        

#         # self.transformation_counts = [] 
        

    
#     def __len__(self):
#         return len(self.folders)
#     #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#     def __getitem__(self, idx):
#         folder_path = self.folders[idx]
#         folder_name = os.path.basename(folder_path)
#         # print("folder_name", folder_name)
#         # print("folder_path", folder_path)
#         data_dict = {}
#         #data = {}
        
#         for modality in self.modalities:
#             modality_path = os.path.join(folder_path, f"{folder_name}-{modality}.nii.gz")
#             data_dict[modality]=modality_path
#             # print("modality path is", modality_path)
#             # if os.path.exists(modality_path):
#             #     #image = nib.load(modality_path).get_fdata()
#             #     data_dict[modality]=modality_path
#             #     print("modality path is", modality_path)
#                 #print("data_dict is", data_dict)
#             # else:
#             #     logging.warning(f"Modality file not found: {modality_path}")
    
#             #logging.debug(f"Data before transformations: {data_dict}")

#             # if self.transforms:
#             #     try:
#             #         data = self.transforms(data)
#             #         logging.debug(f"Data after transformations: {data[modality].shape}")
#             #     except Exception as e:
#             #         logging.error(f"Error during transformations: {e}")
        
#         mask_path = os.path.join(folder_path, f"{folder_name}-seg.nii.gz")
#         # print("mask path is", mask_path)
#         data_dict['mask'] = mask_path
#         data_dict['mask_path'] = mask_path
#         # if os.path.exists(mask_path):
#         #     #mask = nib.load(mask_path).get_fdata()
#         #     data_dict['mask'] = mask_path
#         # # else:
#         #     logging.warning(f"Mask file not found: {mask_path}")

#         #logging.debug(f"Data before transformations: {data_dict}")

#         if self.transforms:
#             #print("transformation availabel")
#             data_dict = self.transforms(data_dict)
#         data_dict['case_id'] = folder_name
        
#             # print("Number of transformations:", len(self.transforms.transforms))
#             # transformation_count = self._count_transformations(data_dict, final_data_dict)
#             # self.transformation_counts.append(transformation_count)
#             # print(f"Image {idx}: {transformation_count} transformations applied")
#             # try:
#             #     print("transformation availabel")
#             #     data_dict = self.transforms(data_dict)
#             #     print(f"Data after transformations: {data_dict['mask'].shape}")
#             # except Exception as e:
#             #     print(f"Error during transformations: {e}")
#         # if self.morph_ops:
#         #     # Apply morphological operations to each modality and mask
#         #     for key in data_dict:
#         #         if key=='mask':
                    
#         #             if len(data_dict[key].shape) == 4:  # Check if it's 4D (e.g., multiple channels)
#         #                 data_dict[key] = self.morph_ops.apply_operations(data_dict[key])
#         return data_dict
        
#     # def _count_transformations(self, original, transformed):
#     #     count = 0
#     #     for key in original:
#     #         if key in transformed:
#     #             # Compare shapes or data if necessary
#     #             if original[key] != transformed[key]:  # Ensure comparison is valid for your data
#     #                 count += 1
#     #                 print(f"Transformation detected for {key}")
#     #     return count


#     # def print_transformation_counts(self):
#     #     print("Transformation counts:")
#     #     for idx, count in enumerate(self.transformation_counts):
#     #         print(f"Image {idx}: {count} transformations applied")

# def Dataloading(data_path, crop_size, modalities):
#     print(data_path)
#     print("crop_size", crop_size)
#     # morph_ops = MorphologicalOperations(num_operations=num_morph_ops)
#     train_dataset = BrainTumorDataset(data_path=data_path, modalities=modalities, crop_size=crop_size, split='train')
#     val_dataset = BrainTumorDataset(data_path=data_path, modalities=modalities, crop_size=crop_size, split='val')
#     test_dataset = BrainTumorDataset(data_path=data_path, modalities=modalities, crop_size=crop_size, split='test')
    
#     return train_dataset, val_dataset, test_dataset
    

# if __name__ == "__main__":
#     args = get_args()
#     data_path="../dataset/processed/"
#     crop_size=args.crop_size
#     modalities=args.modalities
    
#     print(data_path)
    
#     train_dataset, val_dataset, test_dataset=Dataloading(data_path, crop_size, modalities)
    
#     # train_dataset = BrainTumorDataset(data_path=args.data_path, modalities=args.modalities, crop_size=args.crop_size, split='train')
#     # val_dataset = BrainTumorDataset(data_path=args.data_path, modalities=args.modalities, crop_size=args.crop_size, split='val')
#     # test_dataset = BrainTumorDataset(data_path=args.data_path, modalities=args.modalities, crop_size=args.crop_size, split='test')

#     # Print out dataset sizes
#     print(f"Training dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")

#     # Create DataLoaders for each split
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

#     # Iterate through a few batches and print shapes
#     for batch_idx, batch_data in enumerate(train_loader):
#         print(f'Training Batch {batch_idx+1}:')
#         #print((batch_data).keys())
#         print(f'Images batch shape of train modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
#         print('Masks train batch shape:', batch_data['mask'].shape)
                
#         if batch_idx >= 2:
#             break

#     for batch_idx, batch_data in enumerate(val_loader):
#         print(f'Validation Batch {batch_idx+1}:')
#         print(f'Images batch shape of val modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
#         print('Masks val batch shape:', batch_data['mask'].shape)
#         if batch_idx >= 2:
#             break

#     for batch_idx, batch_data in enumerate(test_loader):
#         print(f'Test Batch {batch_idx+1}:')
#         print(f'Images batch shape of test modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
#         print('Masks test batch shape:', batch_data['mask'].shape)
#         if batch_idx >= 2:
#             break




#kflod with json
import logging
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from config.configp import get_args
from src.transformations import get_train_transforms, get_val_transforms

import numpy as np
import random

class BrainTumorDataset(Dataset):
    def __init__(self, json_file, fold, crop_size, split='train'):
        self.crop_size = crop_size
        self.split = split
        self.transforms = get_train_transforms(crop_size) if split == 'train' else get_val_transforms(crop_size)
        print("json_file", json_file)
        with open(json_file, 'r') as f:
            all_data = json.load(f)["training"]

        if self.split == 'train':
            self.data = [entry for entry in all_data if entry["fold"] != fold]
        else:
            self.data = [entry for entry in all_data if entry["fold"] == fold]

        # if self.split == 'train':
        #     self.data = [entry for entry in all_data if entry["fold"] == fold]
        # elif self.split == 'val':
        #     self.data = [entry for entry in all_data if entry["fold"] == fold]
        # elif self.split == 'test':
        #     self.data = [entry for entry in all_data if entry["fold"] == fold]   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        data_dict = {}
        # "t1n", "t2w", "t1c", "t2f"
        # Load modality paths with custom naming
        for modality_path in entry["image"]:
            # if "t1.nii.gz" in modality_path or "_t1_aug" in modality_path:
            #     modality_name = "t1n"
            # elif "t2.nii.gz" in modality_path or "_t2_aug" in modality_path:
            #     modality_name = "t2w"
            # elif "t1ce.nii.gz" in modality_path or "_t1ce_aug" in modality_path:
            #     modality_name = "t1c"
            # elif "flair.nii.gz" in modality_path or "_flair_aug" in modality_path:
            #     modality_name = "t2f"

            #Brats 2023
            # print("modality_path", modality_path)
            if "t1n.nii.gz" in modality_path or "_t1_aug" in modality_path:
                modality_name = "t1n"
            elif "t2w.nii.gz" in modality_path or "_t2_aug" in modality_path:
                modality_name = "t2w"
            elif "t1c.nii.gz" in modality_path or "_t1ce_aug" in modality_path:
                modality_name = "t1c"
            elif "t2f.nii.gz" in modality_path or "_flair_aug" in modality_path:
                modality_name = "t2f"
            # else:
            #     modality_name = "unknown"

            data_dict[modality_name] = modality_path

        # data_dict['mask'] = entry['label']
        # data_dict['mask_path'] = entry['label']
        # data_dict['case_id'] = os.path.basename(entry['label']).split('_seg')[0]


        #Brats 2023

        # data_dict['mask'] = entry['mask']
        # data_dict['mask_path'] = entry['mask']
        # data_dict['case_id'] = os.path.basename(entry['mask']).split('-seg')[0]

        
        data_dict['mask'] = entry['mask']
        data_dict['mask_path'] = data_dict['t2f']

        data_dict['case_id'] = os.path.basename(data_dict['t2f']).split('-t2f')[0]
        
        # data_dict['case_id'] = os.path.basename(entry['mask']).split('-seg')[0]

        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict

def Dataloading(json_file, crop_size, modalities, fold):
    print("Loading data from JSON with 5-fold cross-validation")
    train_dataset = BrainTumorDataset(json_file=json_file, fold=fold, crop_size=crop_size, split='train')
    val_dataset = BrainTumorDataset(json_file=json_file, fold=fold, crop_size=crop_size, split='val')
    # test_dataset = BrainTumorDataset(json_file=json_file, fold=fold, crop_size=crop_size, split='test')
    return train_dataset, val_dataset

# def Dataloading(json_file, crop_size, modalities, fold_train, fold_val, fold_test):
#     print("Loading data from JSON with 5-fold cross-validation")
#     train_dataset = BrainTumorDataset(json_file=json_file, fold=fold_train, crop_size=crop_size, split='train')
#     val_dataset = BrainTumorDataset(json_file=json_file, fold=fold_val, crop_size=crop_size, split='val')
#     test_dataset = BrainTumorDataset(json_file=json_file, fold=fold_test, crop_size=crop_size, split='test')
#     return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    args = get_args()
    crop_size = args.crop_size
    fold = args.fold  # Add this in your config file or command-line args
    json_file = "../dataset/folds_reference.json"  # Make sure this path is correct

    train_dataset, val_dataset, test_dataset = Dataloading(json_file, crop_size, fold)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")









#kflod
# import os
# import random
# import numpy as np
# from sklearn.model_selection import KFold
# import torch
# from torch.utils.data import Dataset, DataLoader
# from config.configp import get_args
# from src.transformations import get_train_transforms, get_val_transforms

# class BrainTumorDataset(Dataset):
#     def __init__(self, data_path, modalities, crop_size, fold_idx=None, k_folds=5, split='train'):
#         self.data_path = data_path
#         self.modalities = modalities
#         self.crop_size = crop_size
#         self.split = split
#         self.k_folds = k_folds
#         self.fold_idx = fold_idx
        
#         # Get all folders
#         self.folders = sorted([os.path.join(data_path, folder) for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])
        
#         # Shuffle the folders for randomness
#         random.seed(42)
#         random.shuffle(self.folders)
        
#         # K-fold split logic
#         kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
#         # Split the data into folds
#         fold_splits = list(kf.split(self.folders))
        
#         # Select the current fold based on fold_idx
#         train_idx, val_test_idx = fold_splits[self.fold_idx]
#         val_idx, test_idx = np.split(val_test_idx, [int(0.2 * len(val_test_idx))])  # Split remaining 50% for validation and test
        
#         # Now assign the data based on split type (train/val/test)
#         if self.split == 'train':
#             self.folders = [self.folders[i] for i in train_idx]
#             self.transforms = get_train_transforms(crop_size)
#         elif self.split == 'val':
#             self.folders = [self.folders[i] for i in val_idx]
#             self.transforms = get_val_transforms(crop_size)
#         elif self.split == 'test':
#             self.folders = [self.folders[i] for i in test_idx]
#             self.transforms = get_val_transforms(crop_size)

#     def __len__(self):
#         return len(self.folders)

#     def __getitem__(self, idx):
#         folder_path = self.folders[idx]
#         folder_name = os.path.basename(folder_path)
#         data_dict = {}

#         # Load the modalities
#         for modality in self.modalities:
#             modality_path = os.path.join(folder_path, f"{folder_name}-{modality}.nii.gz")
#             data_dict[modality] = modality_path

#         # Load the mask
#         mask_path = os.path.join(folder_path, f"{folder_name}-seg.nii.gz")
#         data_dict['mask'] = mask_path

#         # Apply transformations if available
#         if self.transforms:
#             data_dict = self.transforms(data_dict)

#         return data_dict

# def Dataloading(data_path, crop_size, modalities, k_folds=5):
#     # Prepare lists to store datasets for each fold
#     datasets = {'train': [], 'val': [], 'test': []}

#     for fold_idx in range(k_folds):
#         # Create dataset instances for each fold
#         train_dataset = BrainTumorDataset(data_path=data_path, modalities=modalities, crop_size=crop_size, fold_idx=fold_idx, k_folds=k_folds, split='train')
#         val_dataset = BrainTumorDataset(data_path=data_path, modalities=modalities, crop_size=crop_size, fold_idx=fold_idx, k_folds=k_folds, split='val')
#         test_dataset = BrainTumorDataset(data_path=data_path, modalities=modalities, crop_size=crop_size, fold_idx=fold_idx, k_folds=k_folds, split='test')
        
#         datasets['train'].append(train_dataset)
#         datasets['val'].append(val_dataset)
#         datasets['test'].append(test_dataset)
    
#     return datasets
    
# if __name__ == "__main__":
#     args = get_args()
#     data_path = "../dataset/processed/"
#     crop_size = args.crop_size
#     modalities = args.modalities
    
#     # Get datasets for k-fold cross-validation
#     datasets = Dataloading(data_path, crop_size, modalities)
    
#     for fold in range(5):  # Example for 5-fold cross-validation
#         print(f"Processing fold {fold + 1}")
        
#         # Get data for the current fold
#         train_dataset = datasets['train'][fold]
#         val_dataset = datasets['val'][fold]
#         test_dataset = datasets['test'][fold]
        
#         # Print out dataset sizes
#         print(f"Training dataset size (fold {fold+1}): {len(train_dataset)}")
#         print(f"Validation dataset size (fold {fold+1}): {len(val_dataset)}")
#         print(f"Test dataset size (fold {fold+1}): {len(test_dataset)}")

#         # Create DataLoaders for each split
#         train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
#         val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
#         test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

#         # Iterate through a few batches and print shapes
#         for batch_idx, batch_data in enumerate(train_loader):
#             print(f'Training Batch {batch_idx + 1}:')
#             print(f'Images batch shape of train modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
#             print('Masks train batch shape:', batch_data['mask'].shape)
#             if batch_idx >= 2:
#                 break

#         for batch_idx, batch_data in enumerate(val_loader):
#             print(f'Validation Batch {batch_idx + 1}:')
#             print(f'Images batch shape of val modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
#             print('Masks val batch shape:', batch_data['mask'].shape)
#             if batch_idx >= 2:
#                 break

#         for batch_idx, batch_data in enumerate(test_loader):
#             print(f'Test Batch {batch_idx + 1}:')
#             print(f'Images batch shape of test modality:', [batch_data[key].shape for key in ['t1', 't2', 't1ce', 'flair']])
#             print('Masks test batch shape:', batch_data['mask'].shape)
#             if batch_idx >= 2:
#                 break
