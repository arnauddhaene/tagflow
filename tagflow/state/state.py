from .object import StateObject, Image, Reference, Deformation, RoI, Strain

import streamlit as st


class SessionState():

    def __init__(self):
        self.image: Image = Image()
        self.reference: Reference = Reference()
        self.deformation: Deformation = Deformation()
        self.roi: RoI = RoI()
        self.strain: Strain = Strain()

    def clear(self):
        st.legacy_caching.clear_cache()

        for obj in vars(self).keys():
            if issubclass(type(obj), StateObject):
                obj.clear()
