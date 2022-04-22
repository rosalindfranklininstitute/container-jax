import os, sys, time
import subprocess
import logging

# Decode the SLURM job information and the MPI rank of this task instance
JOB_ID, JOB_NAME = os.environ['SLURM_JOB_ID'], os.environ['SLURM_JOB_NAME']
LOCAL_RANK, LOCAL_SIZE = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']), int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
WORLD_RANK, WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_RANK']), int(os.environ['OMPI_COMM_WORLD_SIZE'])

class MPIFilter(logging.Filter):
    # Filter to inject MPI rank and job information into logging output
    def filter(self, record):
        record.job_id, record.job_name = JOB_ID, JOB_NAME
        record.local_rank, record.local_size = LOCAL_RANK, LOCAL_SIZE
        record.world_rank, record.world_size = WORLD_RANK, WORLD_SIZE
        return True

# Create a unique logger for this SLURM job for this MPI rank
logger = logging.getLogger(f'{JOB_ID}-{JOB_NAME}-{WORLD_RANK:04d}')
logger.setLevel(logging.DEBUG)
logger.addFilter(MPIFilter())

# Format the logging to include the SLURM job and MPI ranks
formatter = logging.Formatter('%(asctime)s | %(world_rank)03d:%(world_size)03d | %(local_rank)03d:%(local_size)03d | '
                              '%(job_id)s | %(job_name)s | %(levelname)10s | %(message)s')

# Mirror logging to stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# Mirror logging to unique output file for this SLURM job and MPI rank
file_handler = logging.FileHandler(f'logs/{JOB_ID}-{JOB_NAME}-{WORLD_RANK:04d}.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

########################################################################################################################

try:
    logger.debug('Starting...')

    def nvidia_smi():
        # Utility to get the output of nvidia-smi
        proc = subprocess.Popen(['/usr/bin/nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc.communicate()[0].decode("utf-8").strip()

    # Query the GPUs that are visible to this process
    CUDA_VISIBLE_DEVICES = [] if ('CUDA_VISIBLE_DEVICES' not in os.environ) else \
                           os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    logger.info(f'CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES}')

    # Debug environment for this task instance
    logger.debug(f'PATH {os.environ.get("PATH", "")}')
    logger.debug(f'LD_LIBRARY_PATH {os.environ.get("LD_LIBRARY_PATH", "")}')

    ####################################################################################################################

    import numpy as np

    from mpi4py import MPI
    MPI_COMM_WORLD = MPI.COMM_WORLD

    # Log nvidia-smi to ensure task instances got the correct GPU bindings
    MPI_COMM_WORLD.barrier()
    for line in nvidia_smi().split('\n'):
        logger.info(f'nvidia-smi | {line}')

    # Do a broadcast from each node to the other nodes to test communication
    MPI.COMM_WORLD.barrier()
    for idx in range(MPI.COMM_WORLD.Get_size()):

        xs = -np.ones((MPI.COMM_WORLD.Get_size(),))
        if idx == MPI.COMM_WORLD.Get_rank():
            xs[:] = idx
            logger.info(f'BCAST ROOT {xs}')

        MPI.COMM_WORLD.barrier()
        logger.info(f'BEFORE BCAST {xs}')

        xs = MPI.COMM_WORLD.bcast(xs, root=idx)

        logger.info(f' AFTER BCAST {xs}')

except Exception as ex:
    # Catch any top level exceptions and ensure they are logged
    logger.exception(ex, exc_info=True)

# Keep process alive for a bit at the end of the job to ensure nvidia-smi process binding is reported accurately
time.sleep(10)
logger.debug('Halting...')
