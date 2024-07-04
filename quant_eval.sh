#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu            ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name quant_eval           # give job unique name
#SBATCH --output ./run_files/training-%j.out      # terminal output
#SBATCH --error ./run_files/training-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user sebastian.bieringer@desy.de
#SBATCH --constraint=A100
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
python3 quant_eval.py --approximate_gaussian_inference=$1 --c_factor=$2 --long=$3 --scaled=$4 --start=$5 --stop=$6 --n_stat=$7 --linear=$8