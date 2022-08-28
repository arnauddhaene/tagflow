from typing import Any

import numpy as np

import streamlit as st


def _validate_ndarray(value: np.ndarray) -> np.ndarray:

    if not isinstance(value, np.ndarray):
        raise ValueError(f'Must pass numpy.ndarray. Got {type(value)}')

    if np.isnan(value).sum() > 0:
        raise ValueError('Must pass array that does not contain any NaNs.')
    
    return value


class StateObject():

    def __init__(self, key: str, value: Any = None):
        self.key: str = key
        if self.key not in st.session_state:
            self.update(value)

    def value(self) -> Any:
        if self.key not in st.session_state:
            raise ValueError(f'{self.key} does not exist in the current Streamlit session state.')
        return st.session_state[self.key]

    def update(self, value: Any) -> None:
        if value is not None:
            value = self.__class__._validate(value)
        st.session_state[self.key] = value

    def clear(self):
        st.session_state[self.key] = None

    @staticmethod
    def _validate(value: Any) -> Any:
        raise NotImplementedError()


class Image(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('image', value)

    @staticmethod
    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 3 or not value.shape[0] == 25:
            raise ValueError(f'Must pass array of shape (25, W, H). Got {value.shape}.')

        if not value.dtype == np.uint8:
            value = (value - value.min()) / (value.max() - value.min()) * 255
            value = value.astype(np.uint8)
            
        return value


class Reference(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('reference', value)

    @staticmethod
    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 2 or not value.shape[1] == 2:
            raise ValueError(f'Must pass array of shape (N, 2). Got {value.shape}.')

        if not value.dtype == np.float16:
            value = value.astype(np.float16)

        return value


class Deformation(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('deformation', value)

    @staticmethod
    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 3 or not value.shape[0] == 25 or not value.shape[2] == 2:
            raise ValueError(f'Must pass array of shape (25, N, 2). Got {value.shape}.')

        if not value.dtype == np.float16:
            value = value.astype(np.float16)

        return value


class RoI(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('roi', value)

    @staticmethod
    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 2:
            raise ValueError(f'Must pass array of shape (W, H). Got {value.shape}.')

        if not value.dtype == bool:
            value = value.astype(bool)

        return value


class Strain(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('strain', value)

    @staticmethod
    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 3 or not value.shape[0] == 25 or not value.shape[1] == 3:
            raise ValueError(f'Must pass array of shape (25, 3, N). Got {value.shape}.')

        if not value.dtype == np.float32:
            value = value.astype(np.float32)

        return value


class Contour(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('contour', value)

    @staticmethod
    def _validate(value: np.ndarray) -> np.ndarray:

        # TODO: fix validation of this data type
        # value = _validate_ndarray(value)

        # if not value.ndim == 3 or not value.shape[0] == 2 or not value.shape[3] == 2:
        #     raise ValueError(f'Must pass array of shape (2, N, 2). Got {value.shape}.')

        # if not value.dtype == np.float64:
        #     value = value.astype(np.float64)

        return value
