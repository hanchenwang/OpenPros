#!/bin/bash
#SBATCH --account=behq-dtai-gh
#SBATCH --job-name=prostate_inv2
#SBATCH --output=/u/pjin3/prostate/jobs/output/prostate_inv2.log
#SBATCH --error=/u/pjin3/prostate/jobs/error/prostate_inv2.err
#SBATCH --partition=ghx4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=180G
#SBATCH --time=02:00:00
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pqj5125@psu.edu

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID : $SLURM_JOB_ID" 
echo "=========================================================="

# debugging flags (optional)
# export NCCL_DEBUG=INFO

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=6533

echo "Master Node: $MASTER"
echo "Slave Node(s): $SLAVES"

cd /u/pjin3/prostate/src

source activate myenv

srun -K1 python -u train.py -n baseline -s debug2 -m FCN4_Deep -mc config/inv.yaml -g1v 1 -g2v 1 --lr 1e-4 -b 128 -j 12 \
  --sync-bn --dist-url tcp://$MASTER:$MASTERPORT --world-size $SLURM_NTASKS -nb 1 -eb 4 -pf 1