import os

if os.environ.get('ENV', 'DEV') == 'PRO':
    from .train import *
else:
    from .test import *

__all__ = ['path', 'model_path', 'configs', 'data']