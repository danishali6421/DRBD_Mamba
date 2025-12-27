#!/bin/bash

# Define arguments
# DATA_PATH="./TrainingData/"
DATA_PATH="/home/danish/data/brats_2023_map"
# FILE_PATH="./Folds_JSON/brats2023_val.json"
FILE_PATH="./Folds_JSON/segmamba_for_train_test_file.json"
# FILE_PATH="./Folds_JSON/IMFUSE_UPDATED_JSON_CM.json"
MODALITIES="t1n t2w t1c t2f"
CROP_SIZE="160 160 144"
BATCH_SIZE=3
NUM_WORKERS=8
RESUME="--resume"  # Add this flag if you want to resume training; remove it if you are starting from scratch
VQVAETRAINING="--vqvae_training"
LDMTRAINING="--ldmtraining"
CHECKPOINT_DIR="./model/checkpoints"  # Directory for saving and loading checkpoints
VQVAE="--VQVAE"
LDM="--LDM"
COND="--COND"
CONDTRAINING="--cond_training"
LMUNET="--LMUNET"
LMUNETTRAINING="--lmunet_training"

# Run the Python script with the arguments
    python main.py \
    --data_path $DATA_PATH \
    --file_path $FILE_PATH \
    --modalities $MODALITIES \
    --crop_size $CROP_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir $CHECKPOINT_DIR \
    $VQVAE \
    # $VQVAETRAINING \
    # $RESUME \
    # # #--checkpoint_dir $CHECKPOINT_DIR
