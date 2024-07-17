from typing import *
import os
from copy import deepcopy

from torch.utils import data


class BaseLazyDataset:
    def __init__(
            self, data_args: Iterable = None, data_kwargs: Optional[Dict[str, Any]] = None, split: Optional[str] = None,
    ):
        self._data_args, self._data_kwargs, self._split = data_args, data_kwargs, split
        if data_kwargs is None:
            self._data_kwargs = {}
        from datasets import DatasetDict
        self._data: Optional[DatasetDict] = None

    def init(self):
        if self._data is None:
            self._prepare_data()

    @property
    def data(self):
        if self._data is None:
            self._prepare_data()
        return self._data

    def clean(self):
        self._data = None

    def _get_data(self):
        from datasets import load_dataset, load_from_disk
        if self._data is not None:
            return self._data
        if os.path.exists(self._data_args[0]):
            args = deepcopy(list(self._data_args))
            if self._split is not None:
                args[0] = os.path.join(args[0], self._split)
            return load_from_disk(*args)
        else:
            return load_dataset(*self._data_args, **self._data_kwargs, split=self._split)

    def _prepare_data(self):
        if self._data_args is not None and self._data is None:
            self._data = self._get_data()


class IterableLazyDataset(BaseLazyDataset, data.IterableDataset):
    pass


class LazyDataset(BaseLazyDataset, data.Dataset):
    def __init__(
            self, data_args: Optional[Iterable] = None, data_kwargs: Optional[Dict[str, Any]] = None,
            split: Optional[str] = None, data_length: Optional[int] = None,
    ):
        super().__init__(data_args, data_kwargs, split)
        self._data_length = data_length

    def __len__(self):
        if self._data_length is not None:
            if self._data is not None:
                return min(len(self._data), self._data_length)
            return self._data_length
        elif self._data is not None:
            return len(self._data)
        else:
            tmp_data = self._get_data()
            self._data_length = len(tmp_data)
            return self._data_length
