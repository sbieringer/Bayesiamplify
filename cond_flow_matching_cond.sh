#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu            ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=72:00:00
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
#python3 cond_flow_matching_cond.py --approximate_gaussian_inference=$1 --n_points=$2 --n_rep=$3 --k=$4 #--gamma_scale=$4 --r_mean=$5 #USE WITH APROXIMATE INFERENCE

python3 cond_flow_matching_cond.py --approximate_gaussian_inference=$1 --n_points=$2 --n_rep=$3 --inv_temp=$4 --lr=$5 --noise=$6 --sigma_adam_dir_denom=$7 #--gamma_scale=$4 #USE WITH MCMC