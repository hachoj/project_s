#!/bin/bash 
 
#SBATCH --job-name=SHORTtrain_resunet_nolinres
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=14
#SBATCH --mem=256gb 
#SBATCH --account=weishao 
#SBATCH --qos=weishao 
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

#SBATCH --output=logs/SHORTtrain_resunet_nolinres.log

#SBATCH --time=1:00:00 # example 8 hrs 
 
#SBATCH --mail-user=7023555729@tmomail.net 
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-type=REQUEUE,INVALID_DEPEND,STAGE_OUT 
#SBATCH --mail-type=TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50 
 
hostname;date;pwd 
export XDG_RUNTIME_DIR=${SLURM_TMPDIR} 

module load mamba
mamba activate mus
python scripts/train.py --decoder resunet --batch-size 192 --no-linres