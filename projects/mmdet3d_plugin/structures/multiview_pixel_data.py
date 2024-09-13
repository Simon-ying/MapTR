# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Sequence, Union

import numpy as np
import torch

from mmengine.structures import BaseDataElement


class MultiViewPixelData(BaseDataElement):
    """Data structure for multi-view pixel-level annotations or predictions.

    All data items in ``data_fields`` of ``MultiViewPixelData`` meet the following
    requirements:

    - They all have 4 dimensions in orders of view, channel, height, and width.
    - They should have the same height and width.
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``MultiViewPixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `MultiViewPixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `MultiViewPixelData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can not set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    'The height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    'not consistent with '
                    'the shape of this '
                    ':obj:`PixelData` '
                    f'{self.shape}')
            assert value.ndim in [
                2, 3, 4
            ], f'The dim of value must be 2, 3 or 4, but got {value.ndim}'
            if value.ndim == 2:
                value = value[None, None]
                warnings.warn('The shape of value will convert from '
                              f'{value.shape[-2:]} to {value.shape}')
            elif value.ndim == 3:
                value = value[None]
                warnings.warn('The shape of value will convert from '
                              f'{value.shape[-2:]} to {value.shape}')
            super().__setattr__(name, value)

    # TODO torch.Long/bool
    def __getitem__(self, item: Sequence[Union[int, slice]]) -> 'MultiViewPixelData':
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`PixelData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, 'Only support to slice height and width'
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(
                        0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        'The type of element in input must be int or slice, '
                        f'but got {type(single_item)}')
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(
                f'Unsupported type {type(item)} for slicing PixelData')
        return new_data

    @property
    def shape(self):
        """The shape of pixel data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None

    # TODO padding, resize
