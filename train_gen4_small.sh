#!/bin/bash -l
#SBATCH --job-name=dvs_test
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4 # MPI ranks per node
#SBATCH --hint=nomultithread 
#SBATCH --chdir=/leonardo/home/userexternal/ychen004/ssm-event-gen4
#SBATCH --output=/leonardo/home/userexternal/ychen004/ssm-event-gen4/logs/dvs_train_gen4_small_1.out
#SBATCH --error=/leonardo/home/userexternal/ychen004/ssm-event-gen4/logs/dvs_train_gen4_small_1.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrb_fm-eeg24
ulimit -s unlimited
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPU_IDS=[0,1,2,3]
export BATCH_SIZE_PER_GPU=6
export TRAIN_WORKERS_PER_GPU=3
export EVAL_WORKERS_PER_GPU=1
# export WANDB_API_KEY=787c528c7ea5039fc3327eb30b1e83d0bb58f94f
export WANDB_MODE=offline
export MDL_CFG=small
srun /leonardo/home/userexternal/ychen004/anaconda3/envs/events_signals/bin/python /leonardo/home/userexternal/ychen004/ssm-event-gen4/RVT/train.py model=rnndet dataset=gen4 dataset.path=/leonardo_scratch/fast/IscrB_FM-EEG24/ychen004/gen4 wandb.project_name=ssms_event_cameras \
wandb.group_name=1mpx +experiment/gen4="small.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}


# srun --partition=boost_usr_prod --qos=boost_qos_dbg --time=00:30:00 --nodes=1 --gpus=2 --pty /leonardo/home/userexternal/ychen004/anaconda3/envs/events_signals/bin/python /leonardo/home/userexternal/ychen004/ssms-event-cameras/RVT/train.py model=rnndet dataset=gen4 dataset.path=/leonardo_scratch/fast/IscrB_FM-EEG24/ychen004/gen4 wandb.project_name=ssms_event_cameras \
# wandb.group_name=1mpx +experiment/gen4="small.yaml" hardware.gpus=${GPU_IDS} \
# batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
# hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}

# python /leonardo/home/userexternal/ychen004/ssms-event-cameras/RVT/train.py model=rnndet dataset=gen4 dataset.path=/capstor/scratch/cscs/cyujie/dataset/gen4 \
# checkpoint=/capstor/scratch/cscs/cyujie/DVS/checkpoint/gen4_small.ckpt hardware.gpus=0 +experiment/gen4="small.yaml" \
# batch_size.train=1 batch_size.eval=1 \
# hardware.num_workers.train=0 hardware.num_workers.eval=0

# python /leonardo/home/userexternal/ychen004/ssms-event-cameras/RVT/validation.py dataset=gen4 dataset.path=/capstor/scratch/cscs/cyujie/dataset/gen4 \
# checkpoint=/capstor/scratch/cscs/cyujie/DVS/checkpoint/gen4_small.ckpt use_test_set=1 hardware.gpus=0 +experiment/gen4="small.yaml" \
# batch_size.eval=12 model.postprocess.confidence_threshold=0.001

# srun --environment=dvs python /leonardo/home/userexternal/ychen004/ssms-event-cameras/RVT/train.py model=rnndet dataset=gen4 dataset.path=/capstor/scratch/cscs/cyujie/dataset/gen4 wandb.project_name=ssms_event_cameras \
# wandb.group_name=1mpx +experiment/gen4="small.yaml" hardware.gpus=[0,1,2,3] \
# batch_size.train=3 batch_size.eval=3 \
# hardware.num_workers.train=12 hardware.num_workers.eval=4