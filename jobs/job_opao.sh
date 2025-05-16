#!/bin/bash
#SBATCH --account=behx-dtai-gh
#SBATCH --job-name=prostate_opao
#SBATCH --output=/u/pjin3/prostate/jobs/output/prostate_opao.log
#SBATCH --error=/u/pjin3/prostate/jobs/error/prostate_opao.err
#SBATCH --partition=ghx4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=180G
#SBATCH --time=48:00:00
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
# export NCCL_SOCKET_IFNAME=hsn
module load nccl

cd /u/pjin3/prostate/src

source activate myenv

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=$(comm -23 <(seq 29000 30000) <(ss -tan | awk '{print $4}' \
  | cut -d':' -f2 | grep '[0-9]\{1,5\}' | sort | uniq) | shuf | head -n 1) # Find a free port

echo "Master Node: $MASTER"
echo "Master Port: $MASTERPORT"
echo "Slave Node(s): $SLAVES"


echo "Trying to get started"
srun -K1 python -u train.py -n baseline -s opao -m FCN4_Deep -mc config/inv.yaml -g1v 1 -g2v 1 \
  --lr 1e-4 --lr-warmup-epochs 5 -b 128 -j 8  -nb 30 -eb 8 --k 1e9 \
  -t prostate_train_one_patient_out.txt -v prostate_val_one_patient_out.txt \
  --sync-bn --dist-url tcp://$MASTER:$MASTERPORT --world-size $SLURM_NTASKS -r checkpoint.pth