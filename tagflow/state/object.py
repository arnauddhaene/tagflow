from typing import Any

import numpy as np

import streamlit as st


def _validate_ndarray(value: np.ndarray) -> np.ndarray:

    if not isinstance(value, np.ndarray):
        raise ValueError(f'Must pass numpy.ndarray. Got {type(value)}')

    if np.isnan(value).sum() > 0:
        raise ValueError('Must pass image that does not contain any NaNs.')
    
    return value


class StateObject():

    def __init__(self, key: str, value: Any = None):
        self.key: str = key
        if self.key not in st.session_state:
            if value is not None:
                self.__class__._validate(value)
            st.session_state[self.key] = value

    def value(self) -> Any:
        if self.key not in st.session_state:
            raise ValueError(f'{self.key} does not exist in the current Streamlit session state.')
        return st.session_state[self.key]

    def update(self, value: Any) -> None:
        self.__class__._validate(value)
        st.session_state[self.key] = value

    def clear(self):
        st.session_state[self.key] = None

    @staticmethod
    def _validate(value: Any) -> Any:
        raise NotImplementedError()


class Image(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('image', value)

    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 3 or not value.shape[0] == 25:
            raise ValueError(f'Must pass array of shape (25, W, H). Got {value.shape}.')

        if not value.dtype == np.float64:
            value = value.astype(np.float64)


class Reference(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('reference', value)

    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 2 or not value.shape[1] == 2:
            raise ValueError(f'Must pass array of shape (N, 2). Got {value.shape}.')

        if not value.dtype == np.float64:
            value = value.astype(np.float64)


class Deformation(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('deformation', value)

    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 3 or not value.shape[0] == 25 or not value.shape[2] == 2:
            raise ValueError(f'Must pass array of shape (25, N, 2). Got {value.shape}.')

        if not value.dtype == np.float64:
            value = value.astype(np.float64)


class RoI(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('roi', value)

    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 2:
            raise ValueError(f'Must pass array of shape (W, H). Got {value.shape}.')

        if not value.dtype == bool:
            value = value.astype(bool)


class Strain(StateObject):

    def __init__(self, value: np.ndarray = None):
        super().__init__('strain', value)

    def _validate(value: np.ndarray) -> np.ndarray:

        value = _validate_ndarray(value)

        if not value.ndim == 3 or not value.shape[0] == 25 or not value.shape[1] == 3:
            raise ValueError(f'Must pass array of shape (25, 3, N). Got {value.shape}.')

        if not value.dtype == np.float64:
            value = value.astype(np.float64)
