#!/usr/bin/env bash

CHECKPOINT_PATH="$HOME/mus108-final/model.pth"
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
MODEL_TYPE="Cnn14"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py audio_tagging --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="resources/R9_ZSCveAHg_7s.wav" --cuda

DATASET_DIR="$HOME/mus108-final/genres"
WORKSPACE="$HOME/mus108-final"

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --mini_data


CUDA_VISIBLE_DEVICES=3 python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda

#####
MODEL_TYPE="Transfer_Cnn13"
CHECKPOINT_PATH="/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/660000_iterations.pth"
python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$CHECKPOINT_PATH --freeze_base --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --few_shots=10 --random_seed=1000 --resume_iteration=0 --stop_iteration=10000 --cuda

python3 utils/plot_statistics.py 1 --workspace=$WORKSPACE --select=2_cnn13
