from functools import lru_cache
from typing import Dict

import numpy as np
import yaml


class MetadataLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    """OPV2V includes NumPy numeric scalar tags; never use a general object loader."""


def _numeric_dtype(loader, node):
    value = loader.construct_mapping(node, deep=True)
    dtype = np.dtype(value['args'][0])
    if dtype.kind not in 'biuf' or dtype.itemsize > 16:
        raise ValueError('only numeric NumPy metadata scalars are supported')
    if 'state' in value:
        dtype = dtype.newbyteorder(value['state'][1])
    return dtype


def _numeric_scalar(loader, node):
    dtype, value = loader.construct_sequence(node, deep=True)
    if not isinstance(dtype, np.dtype) or not isinstance(value, bytes) or len(value) != dtype.itemsize:
        raise ValueError('invalid NumPy scalar encoding')
    return np.frombuffer(value, dtype=dtype, count=1)[0].item()


MetadataLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                               lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
MetadataLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.dtype', _numeric_dtype)
MetadataLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', _numeric_scalar)
MetadataLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar', _numeric_scalar)


@lru_cache(maxsize=512)
def read_metadata(filename: str) -> Dict:
    with open(filename) as stream:
        return yaml.load(stream, Loader=MetadataLoader)
