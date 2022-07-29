import io
import enum
from typing import List

import numpy as np

import h5py

import streamlit as st

from .object import StateObject, Image, Reference, Deformation, RoI, Strain, Contour
 

class SessionStatus(enum.Enum):
    none = 1
    image = 2
    reference = 3
    deformation = 4
    strain = 5


class SessionState():

    def __init__(self):
        self.image: Image = Image()
        self.reference: Reference = Reference()
        self.deformation: Deformation = Deformation()
        self.roi: RoI = RoI()
        self.strain: Strain = Strain()
        self.contour: Contour = Contour()

    def clear(self, names: List[str] = None):
        st.legacy_caching.clear_cache()

        for name, obj in vars(self).items():
            if names is None or name in names:
                if issubclass(type(obj), StateObject):
                    obj.clear()

    def status(self):
        if self.image.value() is None:
            return SessionStatus.none
        elif self.reference.value() is None or self.roi.value() is None:
            return SessionStatus.image
        elif self.deformation.value() is None:
            return SessionStatus.reference
        elif self.strain.value() is None:
            return SessionStatus.deformation
        else:
            return SessionStatus.strain

    def not_none_attributes(self):
        return [name for name, artefact in self.__dict__.items() if artefact.value() is not None]

    def prepare_bytes(self):
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as f:
            for name, artefact in self.__dict__.items():
                if name != 'contour' and artefact.value() is not None:
                    f[name] = np.array(artefact.value(), dtype=np.float64)

        return bio
