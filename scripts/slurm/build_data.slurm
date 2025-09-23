#!/bin/bash

#SBATCH --job-name=build_data_for_proj_n
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --account=weishao
#SBATCH --qos=weishao

#SBATCH --time=03:00:00 # example 8 hrs
#SBATCH --output=logs/lg_build_data.log

#SBATCH --mail-user=7023555729@tmomail.net
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=REQUEUE,INVALID_DEPEND,STAGE_OUT
#SBATCH --mail-type=TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50

hostname;date;pwd

export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
conda activate mus
python data_build.py
python data_train_test_split.py