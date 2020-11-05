import os

if os.environ.get('ENV', 'DEV') == 'DEV':
    from .test import *
else:
    from .train import *

__all__ = ['path', 'model_path', 'configs', 'data']