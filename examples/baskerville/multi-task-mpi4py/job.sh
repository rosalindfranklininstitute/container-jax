#!/bin/bash
#SBATCH --account ffnr0871-rfi-test
#SBATCH --qos rfi
#SBATCH --time 30
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-task 0
#SBATCH --job-name multi-task-mpi4py
#SBATCH -o logs/%j-%x.out

module purge
module load baskerville
module load bask-apps/live
module load OpenMPI/4.0.5-gcccuda-2020b

set -x

export ROOTDIR="/bask/projects/f/ffnr0871-rfi-test/pje39613"
export SINGULARITY_CACHEDIR="$ROOTDIR/.singularity-cache"

mpirun singularity run --nv docker://quay.io/rosalindfranklininstitute/jax:v0.3.1-baskerville \
                   python multi-task-mpi4py.py
