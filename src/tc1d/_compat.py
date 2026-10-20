# Optional MPI dependencies
# Import MPI libraries if available, otherwise define dummy variables
try:
    from mpi4py import MPI
    from schwimmbad import MPIPool

    HAS_MPI = True
    COMM_WORLD = MPI.COMM_WORLD
except ImportError:
    HAS_MPI = False
    MPI = None
    MPIPool = None
    COMM_WORLD = None
