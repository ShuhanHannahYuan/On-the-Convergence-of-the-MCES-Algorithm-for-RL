#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila,parallel
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=netid@nyu.edu # put your email here if you want emails

#SBATCH --array=0-319 # here the number depends on number of jobs in the array
#SBATCH --output=r_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=r_%A_%a.err

# #SBATCH --gres=gpu:1 # uncomment this line to request for a gpu if your program uses gpu
#SBATCH --constraint=cpu # use this if you want to only use cpu

# the sleep command will help with hpc issues when you have many jobs loading same files
sleep $(( (RANDOM%5) + 1 ))

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3 mesa/19.0.5 llvm/7.0.1
source deactivate
source activate rl

echo ${SLURM_ARRAY_TASK_ID}
python hpc_runner.py --setting ${SLURM_ARRAY_TASK_ID}
