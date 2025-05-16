#!/bin/bash
#SBATCH --account=behx-dtai-gh
#SBATCH --job-name=prostate_test
#SBATCH --output=/u/pjin3/prostate/jobs/output/prostate_test.log
#SBATCH --error=/u/pjin3/prostate/jobs/error/prostate_test.err
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=120G
#SBATCH --time=1:00:00
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

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=29508

echo "Master Node: $MASTER"
echo "Slave Node(s): $SLAVES"

cd /u/pjin3/prostate/src

source activate myenv

python -u test.py -n baseline -s run2 -m FCN4_Deep -mc config/inv.yaml \
  -b 128 -j 8 --k 1e9 -r model_240.pth -v prostate_test.txt --vis -vsu 240 -vb 1 -vsa 20

python -u test.py -n baseline -s opao -m FCN4_Deep -mc config/inv.yaml \
  -b 128 -j 8 --k 1e9 -r model_240.pth -v prostate_test_one_patient_out.txt --vis -vsu 240 -vb 1 -vsa 20

python -u test.py -n baseline -s opro -m FCN4_Deep -mc config/inv.yaml \
  -b 128 -j 8 --k 1e9 -r model_240.pth -v prostate_test_one_prostate_out.txt --vis -vsu 240 -vb 1 -vsa 20

python -u test.py -n baseline -s combined -m FCN4_Deep -mc config/inv.yaml \
  -b 128 -j 8 --k 1e9 -r model_240.pth -v prostate_test_combined.txt --vis -vsu 240 -vb 1 -vsa 20