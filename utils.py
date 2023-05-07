import numpy as np

def print_obj(obj, logger=None):
    if logger is not None:
        logger.info('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
    else:
        print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))






