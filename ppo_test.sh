#!/bin/bash
#SBATCH --job-name=boba-job
#SBATCH --account=msccsit2024
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
#SBATCH --container-image=/home/szhangfa/containers/boba.img
#SBATCH --container-mounts=/home/szhangfa:/home/szhangfa
#SBATCH --container-workdir=/home/szhangfa
#SBATCH --no-container-mount-home
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --container-env=PYXI_DISABLE_DEFAULT_MOUNTS=1
#SBATCH --container-save=/home/szhangfa/containers/boba.img

cd /home/szhangfa/AReaL
bash examples/run_async_ppo.sh