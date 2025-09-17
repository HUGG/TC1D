import importlib.metadata
import sys

if hasattr(sys, '_called_from_test'):
    # Test run, do nothing
    pass
else:
    # Import normally
    from .tc1d import init_params, prep_model

# Versioning
__version__ = importlib.metadata.version("tc1d")
