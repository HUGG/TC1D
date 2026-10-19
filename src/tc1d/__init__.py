
from .tc1d import init_params, prep_model

# Versioning
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("tc1d")
except ImportError:
    __version__ = "dev"
