"""This python file defines common types that are used in the project.
"""
from typing import TypeVar, Sequence, Union
from torch import Tensor
from torch.nn import Module


T_M = Union[Tensor, Module]
T_M_SEQ = Union[T_M, Sequence[T_M]]


def ExtendedSequence(_type):
    """Returns the sequence or single value type of the given type '_type'.

    Args:
        _type (Any): The given type.

    Returns:
        _type_: _description_
    """
    return Union[_type, Sequence[_type]]

