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

    if len(CUDA_VISIBLE_DEVICES) > 0:
        # Assert that there are an even number of GPUs for local tasks on this node
        assert (len(CUDA_VISIBLE_DEVICES) % LOCAL_SIZE) == 0

        # Determine the GPUs that should be used for this local task instance
        GPUS_PER_TASK = len(CUDA_VISIBLE_DEVICES) // LOCAL_SIZE
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES[(LOCAL_RANK*GPUS_PER_TASK):((LOCAL_RANK+1)*GPUS_PER_TASK)]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(CUDA_VISIBLE_DEVICES)
        logger.info(f'Setting CUDA_VISIBLE_DEVICES to {CUDA_VISIBLE_DEVICES}')

    # Debug environment for this task instance
    logger.debug(f'PATH {os.environ.get("PATH", "")}')
    logger.debug(f'LD_LIBRARY_PATH {os.environ.get("LD_LIBRARY_PATH", "")}')

    ####################################################################################################################

    import jax
    import jax.numpy as jnp
    import numpy as np

    # Use cloned JAX communicator exclusively for JAX to ensure no deadlocks from
    # asynchronous execution compared to surrounding mpi4py communications.
    from mpi4py import MPI
    import mpi4jax
    MPI_COMM_WORLD = MPI.COMM_WORLD
    JAX_COMM_WORLD = MPI_COMM_WORLD.Clone()

    # Log nvidia-smi to ensure task instances got the correct GPU bindings
    MPI_COMM_WORLD.barrier()
    for line in nvidia_smi().split('\n'):
        logger.info(f'nvidia-smi | {line}')
    MPI_COMM_WORLD.barrier()

    # TODO add pmap to example function to utilize multiple local GPUs (currently only uses GPU 0)
    # Create a JAX function that will worth with mpi4jax without causing deadlocks
    @jax.jit
    def test_mpi4jax(xs):
       xs_sum, _ = mpi4jax.allreduce(xs, op=MPI.SUM, comm=JAX_COMM_WORLD)
       return xs_sum

    # Create input array for this task instance
    xs = jnp.arange(JAX_COMM_WORLD.Get_size()) + JAX_COMM_WORLD.Get_rank()
    logger.info(f'BEFORE ALL-REDUCE-SUM | xs {xs.device()} {xs.shape} {xs}')

    # Run the JAX function which includes mpi4jax communication
    xs = test_mpi4jax(xs)
    logger.info(f' AFTER ALL-REDUCE-SUM | xs {xs.device()} {xs.shape} {xs}')

    # Log nvidia-smi to ensure task instances got the correct GPU bindings
    MPI_COMM_WORLD.barrier()
    for line in nvidia_smi().split('\n'):
        logger.info(f'nvidia-smi | {line}')
    MPI_COMM_WORLD.barrier()

except Exception as ex:
    # Catch any top level exceptions and ensure they are logged
    logger.exception(ex, exc_info=True)

# Keep process alive for a bit at the end of the job to ensure nvidia-smi process binding is reported accurately
time.sleep(10)
logger.debug('Halting...')
