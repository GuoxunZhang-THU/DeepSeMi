from enum import Enum, auto
from collections import OrderedDict
from typing import Dict, Tuple


class DataDim(Enum):
    BATCH = auto()
    CHANNEL = auto()
    WIDTH = auto()
    HEIGHT = auto()


DIM_CHAR_DICT = {
    DataDim.BATCH: "B",
    DataDim.CHANNEL: "C",
    DataDim.HEIGHT: "H",
    DataDim.WIDTH: "W",
}
""" Enumeration association to char representations.
"""

CHAR_DIM_DICT = dict((v, k) for k, v in DIM_CHAR_DICT.items())
""" Character association to enumeration representations.
"""


def batch(data_format: str) -> str:
    """ Append batching to a format if it is not already there. Assume LHS.
    """
    if DIM_CHAR_DICT[DataDim.BATCH] not in data_format:
        return DIM_CHAR_DICT[DataDim.BATCH] + data_format
    else:
        return data_format


def unbatch(data_format: str) -> str:
    """ Append batching to a format if it is not already there. Assume LHS.
    """
    return data_format.replace(DIM_CHAR_DICT[DataDim.BATCH], "")


class DataFormat:
    BHWC = "BHWC"
    BWHC = "BWHC"
    BCHW = "BCHW"
    BCWH = "BCWH"
    HWC = "HWC"
    WHC = "WHC"
    CHW = "CHW"
    CWH = "CWH"


PIL_FORMAT = DataFormat.CWH
PIL_BATCH_FORMAT = DataFormat.BCWH
""" Formats used by Pillow/PIL.
"""


DATA_FORMAT_INDEX_DIM = {}
""" Storage for pre-defined dimension format dictionaries that map
axis index to dimension type.
"""

DATA_FORMAT_DIM_INDEX = {}
""" Storage for pre-defined dimension format dictionaries that map
dimension type to axis index.
"""


def make_index_dim_dict(data_format: str) -> Dict:
    dim_dict = OrderedDict()
    for i, c in enumerate(data_format):
        dim_dict[i] = CHAR_DIM_DICT[c]
    return dim_dict


def make_dim_index_dict(data_format: str) -> Dict:
    dim_dict = OrderedDict()
    for i, c in enumerate(data_format):
        dim_dict[CHAR_DIM_DICT[c]] = i
    return dim_dict


def add_format(data_format: str):
    global DATA_FORMAT_INDEX_DIM
    DATA_FORMAT_INDEX_DIM[data_format] = make_index_dim_dict(data_format)
    global DATA_FORMAT_DIM_INDEX
    DATA_FORMAT_DIM_INDEX[data_format] = make_dim_index_dict(data_format)


def permute_tuple(cur: str, target: str) -> Tuple[int]:
    assert sorted(cur) == sorted(target)

    # Ensure reference dictionaries exist
    if cur not in DATA_FORMAT_INDEX_DIM:
        add_format(cur)
    if target not in DATA_FORMAT_DIM_INDEX:
        add_format(target)

    dims_cur = DATA_FORMAT_DIM_INDEX[cur]
    dims_target = DATA_FORMAT_DIM_INDEX[target]
    transpose = [dims_cur[target] for target in dims_target.keys()]
    return tuple(transpose)