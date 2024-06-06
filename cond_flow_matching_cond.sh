#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu            ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name baysiamplify           # give job unique name
#SBATCH --output ./run_files/training-%j.out      # terminal output
#SBATCH --error ./run_files/training-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user sebastian.bieringer@desy.de
#SBATCH --constraint=P100
#SBATCH --exclude=max-cmsg007
 
##SBATCH --nodelist=max-cmsg[001-008]         # you can select specific nodes, if necessary

 
#####################
### BASH COMMANDS ###
#####################
 
## examples:
 
# source and load modules (GPU drivers, anaconda, .bashrc, etc)
source ~/.bashrc

# activate your conda environment the job should use
conda activate new_bayesconda_3
 
# go to your folder with your python scripts
cd /home/bierings/Bayesiamplify/

echo $(date +"%Y%m%d_%H%M%S") $SLURM_JOB_ID $SLURM_NODELIST $SLURM_JOB_GPUS  >> cuda_vis_dev.txt

# run
python3 cond_flow_matching_cond.py --n_points=$1 --n_rep=$2 --k=$3 #--gamma_scale=$4 --r_mean=$5
#python3 cond_flow_matching_cond.py --n_points=$1 --n_rep=$2 --inv_temp=$3 --lr=$4 --noise=$5 --sigma_adam_dir_denom=$6 #--gamma_scale=$4