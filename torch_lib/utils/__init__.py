from typing import Dict, List, Union, Tuple
from torch_lib.utils.type import T_M_SEQ, T_M
from torch import Tensor
from torch.nn import Module
import threading
from functools import wraps
from time import time


class Nothing:
    """
    'Nothing' object, different from python 'None'.
    It often comes from getting properties or items that the object does not have, or simply represents a default value.
    'Nothing' allows any attribute-get or method-call operations without throwing Errors, making the program more stable.
    It will show Warnings in the console instead.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self

    def __getattribute__(self, *_):
        return self

    def __getitem__(self, *_):
        return self

    def __setattr__(self, *_):
        pass

    def __setitem__(self, *_):
        pass


def is_nothing(obj):
    """Check whether an object is an instance of 'Nothing'

    Args:
        obj (Any): object

    Returns:
        bool: whether the object is instance of 'Nothing'
    """
    return isinstance(obj, Nothing)


class Base:
    """
    Base class, making its subclasses be able to use '[]' operations(just like python dict).
    Return 'Nothing' if the object does not have the property being retrieved, without throwing Errors.
    What's more, it allows its subclasses assign properties using a dict.
    """

    def from_dict(self, kwargs: Dict):
        """assign properties to the object using a dict.

        Args:
            kwargs (Dict): property dict.
        """
        self.__dict__.update(kwargs)

    def __getattr__(self, *_):
        return Nothing()

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __getattribute__(self, key):
        return super().__getattribute__(key)


def Singleton(cls):
    """
    Decorator that makes decorated classes singleton.
    It makes the creation of the singleton object thread-safe by using double-checked locking.
    """
    _lock = threading.Lock()
    _instance = {}
    
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in _instance:
            with _lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return wrapper


class SingleConst:
    """
    A class that defines a const value that cannot be changed.
    Show Warnings when the value is attempted to be changed, and the change won't actually take effect.
    Allow assigning the value to 'Nothing' when initialized, which means that you will assign the value later.
    Once the value is not 'Nothing', it will never be changed.

    *****
    Note that the class property cannot be changed means every instance of the class shares the same property.
    Not suitable for the property that varies from instance to instance.
    *****
    """

    def __init__(self, value=Nothing()):
        # the default value will refer to the same 'Nothing'.
        super().__init__()
        self.value = value

    def __set__(self, _, value):
        # the value can be changed only when it's 'Nothing'
        if is_nothing(self.value):
            self.value = value
        else:
            # TODO: show warnings
            print('the value cannot be changed.')

    def __get__(self, *_):
        return self.value


class MultiConst:
    """
    Const constraint that varies from instances. Similar to 'SingleConst'.

    *****
    WARNING:
    MultiConst is not in strict mode. Actually, for convenience, MultiConst will create a private property of
    the same name using prefix '_' in the object. Thus it won't be able to prevent the direct assignment to 
    the private property it creates.

    e.g.
    class Example:
        attr = MultiConst()
    
    ex = Example()
    ex.attr = 'a' # OK
    ex.attr = 'b' # FAIL, cannot assign new value to a const value.
    ex.attr # Now the property value is still 'a'.
    ex._attr = 'c' # **** WARNING: OK, because the MultiConst cannot prevent assignment to this property. ****
    ex.attr # Now the property value has become 'c'.
    *****

    *****
    WARNING:
    Do not use MultiConst to a property named already with prefix '_', for MultiConst will create a new property
    with '__' as a prefix, making the attribute unaccessible through '__foo' outside the class.
    *****
    """
    def __init__(self):
        super().__init__()
    
    def __set_name__(self, _, name):
        self.private_name = '_%s' % str(name)

    def __set__(self, instance, value):
        temp = getattr(instance, self.private_name, Nothing())
        if is_nothing(temp):
            setattr(instance, self.private_name, value)
        else:
            # TODO: show warnings
            print('the value cannot be changed')

    def __get__(self, instance, _):
        return getattr(instance, self.private_name, Nothing())


def AccessFilter(cls):
    """
    Access filter that change methods of a class dynamically.
    """
    @wraps(cls)
    def new_getattr(self, key):
        for access_filter in cls.access_filters:
            # Check every access filter to see if there is a defined filter value to return.
            result = access_filter(self, key)
            if is_nothing(result) is False:
                return result
        # If not in the access filters, then check its super class.
        super_obj = super(cls, self)
        if hasattr(super_obj, "__getattr__"):
            # If super class defines __getattr__, then check it.
            return super_obj.__getattr__(key)
        # Else raise AttributeError
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, key))
    
    # A list that contains all the access filters of the class
    cls.access_filters = SingleConst([])
    cls.__getattr__ = new_getattr
    return cls


def AddAccessFilter(func):
    """
    Add an access filter to a specific class
    """
    def wrapper(cls):
        if hasattr(cls, 'access_filters') is False:
            # TODO: Warning.
            print('please use decorator AccessFilter to the class first before using AddAccessFilter.')
        else:
            cls.access_filters.append(func)
        return cls
    return wrapper


def ListAccessFilter(name):
    """
    Forward list operations to some property of the class.
    """
    def extend(arr):
        """Rewrite the extend operation, so that it can append a single item if it is non-list.

        Args:
            arr (list): _description_
        """
        def wrapper(items):
            if isinstance(items, (list, tuple)):
                arr.extend(list(items))
            else:
                arr.append(items)
        return wrapper

    def access_filter(self, key):
        if key in ['append', 'pop', 'clear', 'insert', 'remove']:
            return getattr(getattr(self, name), key)
        elif key == 'extend':
            return extend(getattr(self, name))
        else:
            return Nothing()
    return access_filter


def get_device(obj: Union[Tensor, Module]):
    """Get the device of the model or tensor.

    Args:
        obj (Union[Tensor, Module]): model or tensor

    Returns:
        device: the device
    """
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.device if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.device
    else:
        return None


def get_dtype(obj: T_M):
    """Get the data type of the model or tensor

    Args:
        obj (T_M): model or tensor

    Returns:
        data type: the data type
    """
    if isinstance(obj, Module):
        parameter = next(obj.parameters(), None)
        return parameter.dtype if parameter is not None else None
    elif isinstance(obj, Tensor):
        return obj.dtype
    else:
        return None


def type_cast(obj: T_M_SEQ, device=None, dtype=None) -> Union[Tuple[Tensor, Module], Tensor, Module, None]:
    """Apply type cast to the model or tensor.

    Args:
        obj (T_M_SEQ): tensor, model, list of tensor or list of model
        device ([type], optional): device. Defaults to None.
        dtype ([type], optional): dtype. Defaults to None.

    Returns:
        Union[Tuple[Tensor, Module], Tensor, Module, None]: [description]
    """
    obj = obj if isinstance(obj, (list, tuple)) else ((obj, ) if isinstance(obj, (Tensor, Module)) else obj)
    if isinstance(obj, (list, tuple)) is False:
        return obj
    if device is not None:
        obj = [item.to(device=device) for item in obj]
    if dtype is not None:
        obj = [item.to(dtype=dtype) for item in obj]
    obj = tuple(obj)
    return obj if len(obj) > 1 else obj[0]


def list_take(list_like, index: Union[List[int], Tuple[int], int]):
    """Get item or sub list of the list_like object through index(es).

    Args:
        list_like (_type_): list-like object
        index (Union[List[int], Tuple[int], int]): the index(es) to be taken.

    Returns:
        _type_: single item or list.
    """
    if index is None:
        return Nothing()
    # convert non-list item to list.
    if isinstance(list_like, (list, tuple)) is False:
        list_like = (list_like,)
    
    list_len = len(list_like)
    # take item(s).
    if isinstance(index, int):
        # return nothing if the index is out of bounds.
        return list_like[index] if index < list_len else Nothing()
    elif isinstance(index, (list, tuple)):
        return tuple(list_like[i] if i < list_len else Nothing() for i in index)


class Iter:

    def __init__(self):
        super().__init__()
        self._iter_start = 0
        # the subclasses have to rewrite the __len__ method
        self._iter_end = self.__len__()

    def __iter__(self):
        # reset _iter_start to 0
        self._iter_start = 0
        return self

    def __next__(self):
        if self._iter_start < self._iter_end:
            # the subclasses have to rewrite the __getitem__ method
            item = self[self._iter_start]
            self._iter_start += 1
            return item
        else:
            raise StopIteration


class IterTool(Iter):

    def __init__(self, arr, progress=False, time=False):
        self.arr = arr
        # 'super' method should be after the self.arr is assigned.
        super().__init__()
        self.items = [progress, time]
        self.func_set = [self.progress, self.time]

    def __getitem__(self, key):
        indexes = [index for index, value in enumerate(self.items) if value is True]
        # func set result
        func_set_res = [func() for func in list_take(self.func_set, indexes)]
        item = self.arr[key]
        return item if len(func_set_res) == 0 else (item, *func_set_res)

    def __len__(self):
        return len(self.arr)

    def progress(self):
        return self._iter_start, self.__len__()

    def time(self):
        return time()
